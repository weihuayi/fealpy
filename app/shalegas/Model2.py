
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

from fealpy.decorator import barycentric
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
            'initial pressure': 3e+6, # Pa
            'initial stress': 6.066e+7, # Pa 先不考虑应力的计算 
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
        self.bc = {'displacement': 0.0, 'flux': 0.0}

        
        self.krw = np.array([
           (      0.39,          0),
           (  0.428125,   0.015625),
           (   0.46625,  0.0441942),
           (  0.504375,  0.0811899),
           (    0.5425,      0.125),
           (  0.580625,   0.174693),
           (   0.61875,    0.22964),
           (  0.656875,   0.289379),
           (     0.695,   0.353553),
           (  0.733125,   0.421875),
           (   0.77125,   0.494106),
           (  0.809375,   0.570045),
           (    0.8475,   0.649519),
           (  0.885625,   0.732378),
           (   0.92375,   0.818488),
           (  0.961875,    0.90773),
           (         1,          1)], dtype=np.float64)

        self.krg = np.array([
               (      0.39,        1.6),
               (  0.425625,   0.544638),
               (   0.46125,   0.491093),
               (  0.496875,   0.439427),
               (    0.5325,   0.389711),
               (  0.568125,   0.342027),
               (   0.60375,   0.296464),
               (  0.639375,   0.253125),
               (     0.675,   0.212132),
               (  0.710625,   0.173627),
               (   0.74625,   0.137784),
               (  0.781875,   0.104816),
               (    0.8175,      0.075),
               (  0.853125,  0.0487139),
               (   0.88875,  0.0265165),
               (  0.924375,   0.009375),
               (      0.96,          0),


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
        self.s = self.cspace.function() # 水的饱和度函数 默认为0, 初始水的饱和度为0
        self.u = self.cspace.function(dim=2) # 位移函数
        self.phi = self.pspace.function() # 孔隙度函数, 分片常数

        # 初值
        self.p[:] = model.rock['initial pressure'] 
        self.phi[:] = model.rock['porosity']

        # 常数矩阵

    @barycentric
    def pressure_coefficient(self, bc):

        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']
        co = self.model.oil['compressibility']

        Sw = self.s.value(bc)

        ps = mesh.bc_to_point(bc)
        phi = self.phi.value(ps)

        val = b - phi
        val /= Ks
        val += phi*Sw*cw
        val += phi(1 - Sw)*co # 注意这里的 co 是常数, 在气体情况下 应该依赖于压力
        return val

    @barycentric
    def saturatin_coefficient(self, bc):
        b = self.model.rock['biot'] 
        Ks = self.model.rock['solid grain stiffness']
        cw = self.model.water['compressibility']

        Sw = self.s.value(bc)
        ps = mesh.bc_to_point(bc)
        phi = self.phi.value(ps)

        val = b - phi
        val /= Ks
        val += phi*cw
        val *= Sw

        return val

    def flux_coefficient(self):
        """
        Notes
        -----
        目前假设是常系数
        """
        lo = 0.5
        lw = 0.5






    def get_current_left_matrix(self, 
        



