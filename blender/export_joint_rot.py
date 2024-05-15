import numpy as np
import bpy

save_dir = 'D:/AI/BulletRL/meshes/dog_anim_angles.npy'
skel_name = 'Bones'
record_bones = ['RFLower', 'RFUpper', 'RBLower', 'RBUpper']
parent_idx = [1, -1, 3, -1]
aligned_rot_axis = 'x'


def main():
    skeleton = check_and_load()
    action =skeleton.animation_data.action
    angles = np.zeros([len(record_bones), int(action.frame_range[1] - action.frame_range[0] + 1)])
    start_rot = np.zeros(len(record_bones))
    for f_idx, frame in enumerate(range(int(action.frame_range[0]), int(action.frame_range[1] + 1))):
        bpy.context.scene.frame_set(frame)
        for b_idx, bone_name in enumerate(record_bones):
            pose_bone = skeleton.pose.bones[bone_name]
            rot_mat = pose_bone.matrix_basis
            rot_rad = rot_mat.to_euler()
            rot_rad = rot_rad[aligned_rot_map[aligned_rot_axis]]
            angles[b_idx, f_idx] = rot_rad
            if f_idx == 0:  # 记录初始姿态
                start_rot[b_idx] = rot_rad

    result = {
        'names': record_bones,
        'parent': parent_idx,
        'angles': angles,
        'start_rot': start_rot
    }
    np.save(save_dir, result)


def check_and_load():
    skeleton = bpy.data.objects[skel_name]
    for target_name in record_bones:
        assert target_name in skeleton.pose.bones.keys(), f"Invalid bone name to record: {target_name}"

    assert aligned_rot_axis in ['x', 'y', 'z'], "aligned axis must be x / y / z"
    return skeleton


if __name__ == '__main__':
    aligned_rot_map = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    main()