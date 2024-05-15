import pybullet as p
import pybullet_data as pd


def connect(mode=p.GUI, enable_GUI=True, timestep=1/240.0):
    client_id = p.connect(mode)
    additional_data_path = pd.getDataPath()
    p.setAdditionalSearchPath(additional_data_path)

    # 载入地面模型，useMaximalCoordinates加大坐标刻度可以加快加载
    ground_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    p.changeDynamics(ground_id, -1, client_id, lateralFriction=1.0)

    if mode == p.GUI:
        # 配置中禁用GUI
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # 是否展示GUI的套件
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(enable_GUI))
        # 禁用 tinyrenderer
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    p.setGravity(0, 0, -9.8)
    timestep = timestep
    p.setTimeStep(timestep)
    p.setRealTimeSimulation(0)

    if mode == p.GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return client_id


def disconnect(id):
    p.disconnect(id)