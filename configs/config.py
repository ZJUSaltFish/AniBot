import collections


class Config:
    bot_name = 'test'
    simulation_rounds = 4  # 每一轮走几步
    simulation_seconds = 2  # 每一步时长
    timestep = 1.0 / 100.0  # 每一模拟步的时间跨度
    steps = None
    total_steps = None  # 总的模拟步数
    nodes = 20  # 一段动画使用多少个节点。左右对称的复用一段动画。

    cma_population = 10
    cma_parents = 5
    cma_iterations = 20
    cma_step = 1.0

    # 生成加工图的参数
    cam_curve_nodes = 10
    cam_min_radius = 1.0  # in cm
    cam_max_radius = 4.0

    binding_radius = 1.0  # 肢体上绑定处的力臂
    anchor_dist = 5.5  # 线末端到凸轮轴心的距离





    def update(self):
        self.steps = int(self.simulation_seconds / self.timestep)
        self.total_steps = self.steps * self.simulation_rounds

cfg = Config()
cfg.update()