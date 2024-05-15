import trimesh
import os
import numpy as np
import abc


class Bot(abc.ABC):
    default_skel = {
        'Body', 'RFUpper', 'RFLower', 'LFUpper', 'LFLower', 'RBUpper', 'RBLower', 'LBUpper', 'LBLower'
    }

    def __init__(self, name='Test', urdf_dir='./meshes/test_bot.urdf'):
        super().__init__()
        self.urdf_dir = urdf_dir
        self.name = name
        self.links = {}
        self.joints = {}

    def write_bot(self):
        with open(self.urdf_dir, 'w') as f:
            # header
            self._line(f, '<?xml version="1.0" ?>', 0)

            self._line(f, f'<robot name="{self.name}" xmlns:xacro="http://www.ros.org/wiki/xacro">')
            for link in self.links.values():
                self._line(f, f'<link name="{link.name}">', 1)

                self._line(f, '<inertial>', 2)
                self._line(f, f'<origin rpy="{link.origin_rpy[0]:0=6f} {link.origin_rpy[1]:0=6f} {link.origin_rpy[2]:0=6f}" xyz="{link.origin_xyz[0]:0=6f} {link.origin_xyz[1]:0=6f} {link.origin_xyz[2]:0=6f}"/>', 3)
                self._line(f, f'<mass value="{link.mass:0=6f}"/>', 3)
                self._line(f, f'<inertia ixx="{link.inertia[0]:0=6f}" ixy="{link.inertia[1]:0=6f}" ixz="{link.inertia[2]:0=6f}" iyy="{link.inertia[3]:0=6f}" iyz="{link.inertia[4]:0=6f}" izz="{link.inertia[5]:0=6f}"/>', 3)
                self._line(f, '</inertial>', 2)

                self._line(f, '<visual>', 2)
                self._line(f, '<geometry>', 3)
                self._line(f, f'<mesh filename="package://{link.mesh_file}"/>', 4)
                self._line(f, '</geometry>', 3)
                # self._line(f, f'<origin rpy="{link.origin_rpy[0]:0=6f} {link.origin_rpy[1]:0=6f} {link.origin_rpy[2]:0=6f}" xyz="{link.origin_xyz[0]:0=6f} {link.origin_xyz[1]:0=6f} {link.origin_xyz[2]:0=6f}"/>', 3)
                self._line(f, f'<material name="default">', 3)
                self._line(f, f'<color rgba="{link.color[0]} {link.color[1]} {link.color[2]} {link.color[3]}"/>', 4)
                self._line(f, '</material>', 3)
                self._line(f, '</visual>', 2)

                self._line(f, '<collision>', 2)
                self._line(f, '<geometry>', 3)
                self._line(f, f'<mesh filename="package://{link.mesh_file}"/>', 4)
                self._line(f, '</geometry>', 3)
                # self._line(f, f'<origin rpy="{link.origin_rpy[0]:0=6f} {link.origin_rpy[1]:0=6f} {link.origin_rpy[2]:0=6f}" xyz="{link.origin_xyz[0]:0=6f} {link.origin_xyz[1]:0=6f} {link.origin_xyz[2]:0=6f}"/>', 3)
                self._line(f, f'<material name="default">', 3)
                self._line(f, f'<color rgba="{link.color[0]} {link.color[1]} {link.color[2]} {link.color[3]}"/>', 4)
                self._line(f, '</material>', 3)
                self._line(f, '</collision>', 2)

                self._line(f, '</link>', 1)

            # if link.joint is not None:
            #     joint = link.joint
            for joint in self.joints.values():
                self._line(f, f'<joint name="{joint.name}" type="{joint.type}">', 1)

                self._line(f, f'<origin rpy="{joint.rpy[0]:0=6f} {joint.rpy[1]:0=6f} {joint.rpy[2]:0=6f}" xyz="{joint.xyz[0]:0=6f} {joint.xyz[1]:0=6f} {joint.xyz[2]:0=6f}"/>', 2)
                self._line(f, f'<parent link="{joint.parent_name}"/>', 2)
                self._line(f, f'<child link="{joint.child_name}"/>', 2)
                self._line(f, f'<axis xyz="{joint.axis[0]} {joint.axis[1]} {joint.axis[2]}"/>', 2)
                self._line(f, f'<limit lower="{joint.rot_limit[0]:0=6f}" upper="{joint.rot_limit[1]:0=6f}"/>', 2)

                self._line(f, '</joint>', 1)

            self._line(f, '</robot>', 0)

    def add_link(self, name:str, obj_file:str, material='POM', is_foot=False):
        linkage = Linkage(name=name, file=obj_file, material=material, is_foot=is_foot)
        self.links[name] = linkage
        return linkage

    def add_joint(self, parent_name, child_name, type='revolute',
                  offset=[0,0,0], rpy=[0,0,0], rot_limit=[-1.57 ,1.57], axis=[1,0,0]):
        joint_name = f'{child_name}_{parent_name}'
        joint = Joint(name=joint_name, parent_name=parent_name, child_name=child_name, type=type,
                           offset=offset, rpy=rpy, rot_limit=rot_limit, axis=axis)
        self.joints[joint_name] = joint

    @abc.abstractmethod
    def load_bot(self, fixed=False, base_position=[0, 0, 0.4]):
        pass

    def _line(self, f, string:str, ident=0):
        for i in range(ident):
            f.write('  ')
        f.write(string + '\n')


class Linkage:
    density_mapping = {
        'POM': 1410
    }
    def __init__(self, name, file='meshes/leg1.obj', material='POM', is_foot=False):
        self.name = name
        self.mesh_file = file
        self.material_file = file.split('.')[0] + '.mtl'
        self.color = [0.8, 0.8, 0.8, 1.0]
        self.density = self.density_mapping[material]
        self.mass = self.get_mass()
        self.origin_xyz = self.get_center_of_mass()
        self.origin_rpy = [0, 0, 0]
        self.inertia = self.get_moment_inertia()
        self.is_foot = is_foot

    def get_mass(self):
        mesh = trimesh.load(self.mesh_file)
        mesh.density = self.density
        if isinstance(mesh, trimesh.Scene):
            print("Imported combined mesh: using centroid rather than center of mass")
            raise NotImplementedError
        else:
            return mesh.mass

    def get_center_of_mass(self):
        mesh = trimesh.load(self.mesh_file)
        mesh.density = self.density
        if isinstance(mesh, trimesh.Scene):
            print("Imported combined mesh: using centroid rather than center of mass")
            raise NotImplementedError
        else:
            return mesh.center_mass

    def get_moment_inertia(self):
        mesh = trimesh.load(self.mesh_file)
        mesh.density = self.density
        if isinstance(mesh, trimesh.Scene):
            print("Imported combined mesh: using centroid rather than center of mass")
            raise NotImplementedError
        else:
            inertia = mesh.moment_inertia
            ixx, iyy, izz = inertia[0,0], inertia[1,1], inertia[2,2]
            ixy, iyz, ixz = 0,0,0
            return ixx, ixy, ixz, iyy, iyz, izz


class Joint:
    def __init__(self, name, parent_name, child_name, type='revolute',
                 offset=[0,0,0], rpy=[0,0,0], rot_limit=[-1.57 ,1.57], axis=[1,0,0]):
        self.name = name
        self.parent_name = parent_name
        self.child_name = child_name
        self.type = type
        self.xyz = offset
        self.rpy = rpy
        self.rot_limit = rot_limit
        self.axis = axis






