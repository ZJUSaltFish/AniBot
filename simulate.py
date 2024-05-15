import time
from tqdm import tqdm
import numpy as np
import pybullet as p
import pybullet_data as pdata
from utils.prepare import connect
from utils.anibot import AniBot
from configs.config import cfg
from utils.losses import *
from utils.cma_es import CMA_ES
from scipy.interpolate import make_interp_spline

def construct_bot(cfg):
    bot = AniBot(cfg)
    body = bot.add_link(
        name='Body', obj_file='meshes/dog1_body.obj'
    )
    motor = bot.add_link(
        name='Motor', obj_file='meshes/motor_small.obj'
    )
    lfu = bot.add_link(
        name='LFU', obj_file='meshes/dog1_frontUpperLeg.obj'
    )
    lfl = bot.add_link(
        name='LFL', obj_file='meshes/dog1_frontLowerLeg.obj', is_foot=True
    )
    rfu = bot.add_link(
        name='RFU', obj_file='meshes/dog1_frontUpperLeg.obj'
    )
    rfl = bot.add_link(
        name='RFL', obj_file='meshes/dog1_frontLowerLeg.obj', is_foot=True
    )
    lbu = bot.add_link(
        name='LBU', obj_file='meshes/dog1_backUpperLeg.obj'
    )
    lbl = bot.add_link(
        name='LBL', obj_file='meshes/dog1_backLowerLeg.obj', is_foot=True
    )
    rbu = bot.add_link(
        name='RBU', obj_file='meshes/dog1_backUpperLeg.obj'
    )
    rbl = bot.add_link(
        name='RBL', obj_file='meshes/dog1_backLowerLeg.obj', is_foot=True
    )
    lfu_body = bot.add_joint(
        parent_name='Body', child_name='LFU', offset=[-0.09, 0.15, 0], rpy=[0,0,0], axis=[-1,0,0], rot_limit=[-0.8,-0.8]
    )
    lfl_lfu = bot.add_joint(
        parent_name='LFU', child_name='LFL', offset=[0,0,-0.12], rpy=[0, 0, 0], axis=[-1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    rfu_body = bot.add_joint(
        parent_name='Body', child_name='RFU', offset=[0.09, 0.15, 0], rpy=[0, 0, 0], axis=[1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    rfl_rfu = bot.add_joint(
        parent_name='RFU', child_name='RFL', offset=[0, 0, -0.12], rpy=[0, 0, 0], axis=[1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    lbu_body = bot.add_joint(
        parent_name='Body', child_name='LBU', offset=[-0.09, -0.11, 0], rpy=[0, 0, 0], axis=[-1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    lbl_lbu = bot.add_joint(
        parent_name='LBU', child_name='LBL', offset=[0, 0, -0.12], rpy=[0, 0, 0], axis=[-1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    rbu_body = bot.add_joint(
        parent_name='Body', child_name='RBU', offset=[0.09, -0.11, 0], rpy=[0, 0, 0], axis=[1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    rbl_rbu = bot.add_joint(
        parent_name='RBU', child_name='RBL', offset=[0, 0, -0.12], rpy=[0, 0, 0], axis=[1, 0, 0], rot_limit=[-0.8, -0.8]
    )
    motor_body = bot.add_joint(
        parent_name='Body', child_name='Motor', type='fixed', offset=[0,0, 0.04]
    )
    bot.front_vec = np.array([0, 1, 0])
    bot.write_bot()
    return bot


def simulate_iter(bot, params, client_id):
    """
    模拟一次，得到结果用于优化
    :param bot:
    :param params: dict[params]
    :param client_id:
    :return:
    """
    # 加入bot
    robot_id = bot.load_bot(client_id=client_id, base_position=[0, 0, 1])
    # 设置bot的参数，该参数来自上次优化的结果
    nodes = params['nodes']
    # nodes = params_to_joint_nodes(nodes, [(0,2), (1,3), (4,6), (5,7)])
    params = nodes_to_joint_params(nodes, [(0,2), (1,3), (4,6), (5,7)])
    bot.update_optim_params(params)
    # 准备用于计算的参数
    v_base = []
    pos_base = []
    # 模拟
    print('Simulating...')

    # 首先降落到地上
    for i in range(200):
        p.stepSimulation()
        states = bot.state(client_id)
        if states['land']:
            break
    for step in tqdm(range(int(cfg.total_steps))):
        # 更新step步的状态
        bot.step(step, client_id)
        p.stepSimulation()
        # 获得step+1步的状态
        states = bot.state(client_id)
        v_base.append(states['v_base'])
        pos_base.append(states['pos_base'])
        if states['fall']:
            break
        else:
            time.sleep(cfg.timestep)
    v_base = np.stack(v_base, axis=0)  # [T, 3]
    a_base = v_base[1:] - v_base[:-1]
    pos_base = np.stack(pos_base, axis=0)  # [T, 3]
    losses = {
        'forward': forward_loss(pos_base[0], pos_base[-1], bot.front_vec),
        'upright': upright_loss(step+1, cfg.total_steps) / (float(cfg.total_steps)**2 * 0.1),
        'smooth': smoothness_loss(a_base) * 0.01
    }
    # 清除bot
    p.removeBody(robot_id)
    return losses


def params_to_joint_nodes(param: np.array, lr_map = []):
    """对称和重复
    :param param: [J*N] J为左侧肢体关节按顺序堆叠。
    """
    param = param.reshape(-1, cfg.nodes)
    assert len(param) == len(lr_map)
    # 左右对称：相差1/2相位
    new_param = np.zeros([len(param)*2, cfg.nodes])
    for i, l_r in enumerate(lr_map):
        l_id = l_r[0]
        r_id = l_r[1]
        new_param[l_id] = param[i]
        new_param[r_id, 0:cfg.nodes//2] = param[i, cfg.nodes//2:]
        new_param[r_id, cfg.nodes//2:] = param[i, 0:cfg.nodes//2]
    # 走路动画循环
    nodes = np.concatenate([new_param for i in range(cfg.simulation_rounds)],axis=-1)

    return nodes

def nodes_to_joint_params(nodes: np.array, lr_map=[]):
    nodes = nodes.reshape(-1, cfg.nodes)
    assert len(nodes) == len(lr_map)
    # 循环
    nodes = np.concatenate([nodes for i in range(cfg.simulation_rounds)], axis=-1)  # [J/2, N*n]
    num_steps = cfg.total_steps  # 总模拟步长
    steps = np.array(list(range(0, num_steps)))  # 每一步的列表
    params = np.zeros([nodes.shape[0], num_steps])
    for idx, j_nodes in enumerate(nodes):
        node_xs = np.linspace(0, num_steps - 1, len(j_nodes))  # 把 nodes 均匀分布在 steps 中
        B = make_interp_spline(node_xs, j_nodes)
        params[idx] = B(steps)  # 采样
    # 对称
    new_param = np.zeros([len(params)*2, num_steps])
    for i, l_r in enumerate(lr_map):
        l_id = l_r[0]
        r_id = l_r[1]
        new_param[l_id] = params[i]
        # 右脚和左脚相差1/2相位
        phi = cfg.steps // 2
        new_param[r_id, 0:phi] = params[i, -phi:]
        new_param[r_id, phi:] = params[i, 0:-phi]
    return new_param


def main():
    torque = 1
    bot = construct_bot(cfg)
    optim_client_id = connect(enable_GUI=False)
    cma = CMA_ES(cfg.cma_population, cfg.cma_parents, cfg.nodes * len(bot.revolute_joints)//2, cfg.cma_step)
    zip_params = cma.generate()
    params = [
        {'nodes': zip_params[i]} for i in range(len(zip_params))
    ]
    for iter in range(cfg.cma_iterations):
        losses = []
        print(f"starting iter {iter}...")
        for sample in range(cfg.cma_population):
            loss = simulate_iter(bot, params[sample], optim_client_id)
            ls = 0
            for k, v in loss.items():
                ls += v
            losses.append(ls)
        nodes = [pa['nodes'].reshape(-1) for pa in params]
        nodes = np.stack(nodes, axis=0)
        losses = np.array(losses)  # [J/2, N]
        print(f"losses of each sample: {losses}")
        new_nodes = cma.step(nodes, losses)
        params = [
            {'nodes': new_nodes[i]} for i in range(len(new_nodes))
        ]


if __name__ == '__main__':
    main()