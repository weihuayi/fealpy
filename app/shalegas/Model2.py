
"""

Notes
-----
这是一个两相流和地质力学耦合的模型, 需要求的量有

* p: 压强
* v: 总速度
* S_w: 水的饱和度 S_w
* u: 岩石位移  

目前, 模型
* 忽略了毛细管压强和重力作用
* 没有考虑裂缝

渐近解决方案:
1. Picard 迭代
2. 气的可压性随着压强的变化而变化
3. 考虑渗透率随着孔隙度的变化而变化 
4. 考虑裂缝
"""

import numpy as np

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d



class WaterFloodingModel():

    def __init__(self):
        self.domain=[0, 10, 0, 10] # m
        self.rock = {
            'permeability': 2000, # md
            'porosity': 0.3, # None
            'lame':(1.0e+8, 3.0e+8), # lambda and mu
            'biot': 1.0,
            'initial pressure': 3, # MPa
            'initial stress': 60.66, # MPa
            'solid grain stiffness': 6.25
            }
        self.oil = {
            'viscosity': 1, # cp
            'compressibility': 1.0e-9, # Pa^{-1}
            'initial saturation': 1.0, 
            'production rate': 3.50e-6 # s^{-1}
            }
        self.water = {'viscosity': 2, # cp
            'compressibility': 2.0e-9 # Pa^{-1}
            'initial saturation': 0.0, 
            'injection rate': 3.51e-6 # s^{-1}
            }
        bc = {'displacement': 0.0, 'flux': 0.0}

    def space_mesh(self, n=32):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, T=1, n=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, T, n)
        return timeline

        




class WaterFloodingModelSolver():

    def __init__(self, model):
        self.model = model
        self.mesh = model.space_mesh()
        self.timeline = model.time_mesh()

        self.vspace = RaviartThomasFiniteElementSpace2d(mesh, p=0) # 速度空间
        self.pspace = self.vspace.smspace # 压强空间
        self.cspace = LagrangeFiniteElementSpace(mesh, p=1) # 位移和饱和度连续空间

        self.v = self.vspace.function() # 速度函数
        self.p = self.pspace.function() # 压强函数
        self.s = self.cspace.function() # 饱和度函数
        self.u = self.cspace.function(dim=2) # 位移函数
        self.phi = self.pspace.function() # 孔隙度函数, 分片常数


    def get_current_left_matrix(self, 
        



