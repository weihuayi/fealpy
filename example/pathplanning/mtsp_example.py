from fealpy.backend import backend_manager as bm
from fealpy.pathplanning import TravellingSalesmanProb, MultipleTravelingSalesmanProb
from fealpy.opt import *

# TSP
pos = bm.random.rand(15, 2) * 100 # 城市坐标
text = TravellingSalesmanProb(pos) 
text.opt(N=50, MaxIT=200, opt_alg=QuantumParticleSwarmOpt) # 利用算法求解
text.plot_route(pos) # 输出路径

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

# 城市坐标，其中仓库坐标在最后
data = bm.concatenate([pos, bm.mean(pos, axis=0)[None, :]])

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

text = MultipleTravelingSalesmanProb(4, data, up_opt_dict, down_opt_dict)
text.solver()
text.output_route(text.up_optimizer.gbest)
text.visualization()