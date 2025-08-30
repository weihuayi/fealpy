from fealpy.backend import backend_manager as bm
from fealpy.pathplanning.model import PathPlanningModelManager
from fealpy.opt import QuantumParticleSwarmOpt, SnowAblationOpt

# TSP
pos = bm.random.rand(15, 2) * 100 # 城市坐标
options = {
    'pos': pos,
    'NP': 50,
    'MaxIT': 200,
    'opt_alg': QuantumParticleSwarmOpt
}
manager = PathPlanningModelManager('travelling_salesman_prob')
text = manager.get_example(1, **options)
text.solver() # 利用算法求解
text.visualization(pos) # 输出路径

# mTSP
pos = bm.array([
    [1150, 1760],
    [630, 1660],
    [40, 2090],
    [750, 1100],
    [750, 2030],
    [1030, 2070],
    [1650, 650],
    [1490, 1630],
    [790, 2260],
    [710, 1310],
    [840, 550],
    [1170, 2300],
    [970, 1340],
    [510, 700],
    [750, 900],
    [1280, 1200],
    [230, 590],
    [460, 860],
    [1040, 950],
    [590, 1390],
    [830, 1770],
    [490, 500],
    [1840, 1240],
    [1260, 1500],
    [1280, 790],
    [490, 2130],
    [1460, 1420],
    [1260, 1910],
    [360, 1980],
    [420, 1930]
])

warehouse_pos = bm.mean(pos, axis=0)[None, :]

up_opt_dict = {
    'opt_alg': QuantumParticleSwarmOpt,
    'NP': 50,
    'MaxIT': 500
}
down_opt_dict = {
    'opt_alg': SnowAblationOpt,
    'NP': 50,
    'MaxIT': 500
}

options = {
    'uav_num': 4,
    'pos': pos,
    'warehouse': warehouse_pos,
    'up_opt_dict': up_opt_dict,
    'down_opt_dict': down_opt_dict,
}

manager = PathPlanningModelManager('travelling_salesman_prob')
text = manager.get_example(2, **options)

text.solver()
text.output_route(text.up_optimizer.gbest)
text.visualization()