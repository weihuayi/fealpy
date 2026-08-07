import argparse

options = {
    'backend': 'numpy',
    'max_iter': 1,
    'tolerance': 1e-5,
    'solve': 'direct',
    'method': 'Ossen',
    'run': 'main',
    'maxstep': 1,
    'tol':1e-8,
    'error_com': False,
    'space_degree': 1,
    'E': 2.1e11,  
    'nu': 0.3,
    'rho': 7800,
    'fluid_rho' : 1.0,
    'mu': 0.003,
    'pbar_log': True,
    'log_level': 'INFO',
    'mesh_type': 'uniform_tet'
}

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.15,     # 使用默认网格大小策略
    "mesh_size_bend": 0.15,
    "mesh_size_interface": 0.15,
}

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.mesher import ElbowPipeMesher
from fealpy.fsi.hydraulic_pipe_fsi_model import HydraulicPipeFSIModel
from fealpy.fsi.coupling_fsi_fem_model import  HydraulicPipeFSIFEMModel

mesher = ElbowPipeMesher(params=params)
pde = HydraulicPipeFSIModel(options, mesher=mesher)
model = HydraulicPipeFSIFEMModel(options, pde=pde)
model.run()

print("-----------------------------")