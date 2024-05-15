import numpy as np
from  configs.config import cfg
import scipy
import matplotlib.pyplot as plt

angles_dir = 'D:/AI/BulletRL/meshes/dog_anim_angles.npy'

def process_rotations(data):
    names = data['names']
    parent = data['parent']
    rots = data['angles']
    lengths = rots * cfg.binding_radius  # in cm
    parented_lengths = np.zeros_like(lengths)  # 考虑了父关系
    for idx, name in enumerate(names):
        parented_lengths[idx] += lengths[idx]
        p_idx = parent[idx]
        while p_idx >= 0:
            parented_lengths[idx] += lengths[p_idx]
            p_idx = parent[p_idx]
    # 等分圆周
    c_divs = np.linspace(0, 2 * np.pi, num=parented_lengths.shape[1] + 1)
    c_divs[-1] = c_divs[0]  # 闭环
    cam_contours = []  # 每条腿的驱动凸轮的外轮廓
    cam_radius_samples = []  # 上述外轮廓每帧接触点的采样
    for idx, lens in enumerate(parented_lengths):
        lens = np.concatenate([lens, lens[[0]]], axis=-1)  # 需要的长度
        # 简化版本的计算
        # 张开角度的最小值（x轴正向旋转的最大弧度）对应最小拉伸（固定值）
        # plt.figure()
        # start_frame = np.argmax(lens)
        lens = np.abs(lens - np.max(lens))
        radius = calc_cam_radius_simple(lens)
        xs = radius * np.cos(c_divs)  # [f]
        ys = radius * np.sin(c_divs)  # [f]
        xy = np.stack([xs, ys], axis=1)
        # diff = calc_diff(xy, lens, start_frame)
        tck, u = scipy.interpolate.splprep([xs, ys], s=0, per=True)
        cam_contours.append(tck)  # 构成外边缘的B样条曲线
        cam_radius_samples.append(radius)
        # t = np.linspace(0, 1, 1000)
        # xi, yi = scipy.interpolate.splev(t, tck)
    return cam_contours, cam_radius_samples
        # plt.plot(xs, ys, 'o', label='原始数据')
        # plt.plot(xi, yi, '-', label='B样条曲线')
        # plt.legend()
        # plt.show()


def calc_cam_radius_simple(length):
    """
    只考虑顶点接触时的驱动计算。从需要的拉伸量计算凸轮的半径
    :param length:
    :return:
    """
    len0 = np.sqrt(cfg.anchor_dist**2 + cfg.cam_min_radius**2)
    r = np.sqrt((length * 0.5 + len0)**2 - cfg.anchor_dist**2)
    return r

def calc_diff(points, target_length, base_frame=None):
    """
    计算当前凸轮外轮廓的驱动效果和实际需要的驱动效果的差异
    我们假设y=0，x>0处是凸轮的接触点。凸轮是t=0时的，默认每一帧转过一个坐标。所以每一帧刚好对应第0帧时的某个点。
    同时，假设在y轴正负半轴等距处有两个锚点。
    :param points: np.array([frame, 2])  外轮廓采样点，x坐标.
    :param target_length: 目标驱动长度
    :param base_frame: 驱动过程中的长度最小值（目标驱动长度为0的那一帧）
    :return:
    """
    # 对每一帧计算凸包，用沿凸包路径近似实际的长度，减去目标长度作为差值。
    if base_frame is None:
        base_frame = np.where(base_frame == 0)[0][0]
    num_frames = len(points)
    for frame in range(num_frames):
        now_points = np.concatenate([points[frame:], points[:frame]], axis=0)
        valid_points = now_points[now_points[:, 0] > 0]
        anchors = np.array([[0, cfg.anchor_dist], [0, -cfg.anchor_dist]])
        hull_points = np.concatenate([valid_points, anchors], axis=0)
        hull = scipy.spatial.ConvexHull(hull_points)
        vert_points = hull_points[hull.vertices]  # 凸包的顶点
        if frame == 0:
            vert_points = np.concatenate([vert_points, vert_points[[0]]], axis=0)
            plt.plot(vert_points[:, 0], vert_points[:, 1], 'r--', label='凸包')
            break
        pass


if __name__ == '__main__':
    data = np.load(angles_dir, allow_pickle=True).item()
    cam_contours, cam_radius_samples = process_rotations(data)
    pass