from fealpy.pathplanning import WeightTargetsSweepCoverageAlg

planner = WeightTargetsSweepCoverageAlg()
planner.set_base(25, 25) 

# 添加目标点
# planner.add_target(10, 10, weight=5)   # 目标点1
# planner.add_target(30, 15, weight=8)   # 目标点2
# planner.add_target(15, 30, weight=3)   # 目标点3
# planner.add_target(35, 35, weight=10)  # 目标点4
# 可以继续添加更多目标点...

# 或者生成随机场景
planner.generate_random_scenario(200, 50, 1, 10)

# 设置无人机参数并执行规划
results = planner.plan_paths(
    M=5,              # 无人机数量
    V=72.0,           # 速度 (km/h)
    omega=0.1,         # 角速度 (rad/s)
    T_fmax=90,         # 最大飞行时间 (min)
    T_s=8,             # 单架无人机设置时间 (min)
    O=1,               # 操作员数量
    alpha=0.5          # 平衡因子
)
# 打印结果摘要
planner.print_summary()

# 可视化路径
planner.plot_paths(area_size=50)
