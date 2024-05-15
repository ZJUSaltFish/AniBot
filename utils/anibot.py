import os
import numpy as np
import pybullet as p
import pybullet_data as pd
from utils.base_bot import Bot, Linkage
import collections
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


class AniBot(Bot):
    def __init__(self, cfg):
        super().__init__(name=cfg.bot_name, urdf_dir=f'meshes/{cfg.bot_name}.urdf')
        self.timestep = cfg.timestep  # 每步时长
        self.num_steps = cfg.steps * cfg.simulation_rounds  # 总步数
        self.steps = np.array(list(range(0, self.num_steps)))
        self.num_nodes = cfg.nodes * cfg.simulation_rounds  # 总节点数 = 动画的节点数 X 播放轮次
        self.node_xs = np.linspace(0, self.num_steps-1, self.num_nodes)  # 所有的x节点，包含起止
        self.front_vec = np.array([0, 1, 0])
        # 这些参数要写入urdf之后才可用
        self.revolute_joints = []
        self.link_names = []
        # 优化参数
        self.nodes = {}  # 一组y坐标，定义了B-spline的插值点 shape= J, N
        self.torques = {}  # shape= [J, num_steps]
        self.bot_id = 0

    # def update_optim_params(self, nodes):
    #     """
    #     更新B样条的插值点，进而更新torques
    #     :param: nodes: ndarray [J, N] of nodes' y coord
    #     :return: None
    #     """
    #     nodes = nodes.reshape(len(self.revolute_joints), self.num_nodes)
    #     for idx, j_name in enumerate(self.revolute_joints):
    #         self.nodes[j_name] = nodes[idx]
    #     for j_name in self.revolute_joints:
    #         xs = self.node_xs
    #         ys = self.nodes[j_name]
    #         B = make_interp_spline(xs, ys)
    #         self.torques[j_name] = B(self.steps)  # sample B-spline
    #         # plt.plot(self.steps, self.torques[j_name])
    #         # pass
    def update_optim_params(self, params):
        for idx, j_name in enumerate(self.revolute_joints):
            self.torques[j_name] = params[idx]

    def step(self, timestep, client_id=0):
        """
        模拟更新一步的物理量
        :param timestep:
        :param client_id:
        :return:
        """
        for i in range(len(self.joints.keys())):
            joint_index, joint_name, joint_type, q_index, u_index, flags, joint_damping, joint_friction, \
            joint_lower_limit, joint_upper_limit, joint_max_force, joint_max_velocity, link_name, joint_axis, \
            parent_frame_pos, parent_frame_orient, parent_index = p.getJointInfo(self.bot_id, i, client_id)
            if joint_type != p.JOINT_REVOLUTE:
                continue
            torque = self.torques[joint_name.decode('utf-8')][timestep]
            p.setJointMotorControl2(self.bot_id, joint_index, p.VELOCITY_CONTROL, targetVelocity=torque)
            # p.setJointMotorControl2(self.bot_id, joint_index, p.POSITION_CONTROL, targetPosition=torque)
            # p.setJointMotorControl2(self.bot_id, joint_index, p.TORQUE_CONTROL, force=torque)

    def state(self, client_id=0):
        """
        获得当前的机器人状态
        :param client_id:
        :return:
        """
        is_fall = False
        is_land = False
        for i in range(len(self.joints)):
            joint_index, joint_name, joint_type, q_index, u_index, flags, joint_damping, joint_friction, \
                joint_lower_limit, joint_upper_limit, joint_max_force, joint_max_velocity, link_name, joint_axis, \
                parent_frame_pos, parent_frame_orient, parent_index = p.getJointInfo(self.bot_id, i, client_id)
        #     world_pos, world_orient, local_inertia_frame_pos, local_inertia_frame_orient, world_link_frame_pos,\
        #         world_link_frame_orient, world_link_velo, world_link_angular_v = \
        #         p.getLinkState(self.bot_id, i, physicsClientId=client_id, computeLinkVelocity=True)
            link = self.links[link_name.decode('utf-8')]
            contact_points = p.getContactPoints(bodyA=self.bot_id, linkIndexA=i, physicsClientId=client_id)
            if len(contact_points) != 0 and not link.is_foot:
                for cp in contact_points:
                    if cp[1] != cp[2]:  # 自碰撞不算，碰到别的才算
                        is_fall = True
            if len(contact_points) != 0 and link.is_foot:
                is_land = True

        v_base, ang_v_base = p.getBaseVelocity(self.bot_id, client_id)
        pos_base, orient_base = p.getBasePositionAndOrientation(self.bot_id, client_id)  # xyz and quaternion
        return {
            'v_base': np.array(v_base),
            'pos_base': np.array(pos_base),
            'orient_base': np.array(orient_base),
            'fall': is_fall,
            'land': is_land
        }

    def write_bot(self):
        super().write_bot()
        for name, joint in self.joints.items():
            if joint.type == 'revolute':
                self.revolute_joints.append(joint.name)
        self._prepare_torques()

    def load_bot(self, client_id=0, fixed=False, base_position=[0, 0, 0.4]):
        self.bot_id = p.loadURDF(self.urdf_dir, useFixedBase=fixed, basePosition=base_position, physicsClientId=client_id)
        # self._set_friction(client_id)
        return self.bot_id

    def _prepare_torques(self):
        for name in self.revolute_joints:
            self.nodes[name] = np.random.randn(self.num_nodes) * 0.001
            self.torques[name] = np.zeros(self.num_steps)

    def _set_friction(self, client_id):
        for i in range(len(self.joints)):
            p.changeDynamics(self.bot_id, i, client_id, lateralFriction=0.0)

