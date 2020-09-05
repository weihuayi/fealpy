
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


Data
----

Coupling of Stress Dependent Relative Permeability and Reservoir Simulation

Water-oil relative permeability SWT
 SW        krw      krow
0.200000 0.000000 0.510200
0.235714 0.000010 0.439917
0.271429 0.000163 0.374841
0.307143 0.000824 0.314970
0.342857 0.002603 0.260306
0.378571 0.006355 0.210848
0.414286 0.013177 0.166596
0.450000 0.024412 0.127550
0.485714 0.041647 0.093710
0.521429 0.066710 0.065077
0.557143 0.101676 0.041649
0.592857 0.148864 0.023428
0.628571 0.210836 0.010412
0.664286 0.290398 0.002603
0.700000 0.390600 0.000000

Liquid-Gas Relative Permeability SLT

 SL        krg      krog
0.500000 0.477200 0.000000
0.535714 0.411463 0.002603
0.571429 0.350596 0.010412
0.607143 0.294598 0.023428
0.642857 0.243469 0.041649
0.678571 0.197210 0.065077
0.714286 0.155820 0.093710
0.750000 0.119300 0.127550
0.785714 0.087649 0.166596
0.821429 0.060867 0.210848
0.857143 0.038955 0.260306
0.892857 0.021912 0.314970
0.928571 0.009739 0.374841
0.964286 0.002435 0.439917
1.000000 0.000000 0.510200


https://www.coursehero.com/file/14152920/Rel-Perm/ 

 np.array([[0.2     , 0.      , 0.8     ],
           [0.225   , 0.001172, 0.703125],
           [0.25    , 0.004688, 0.6125  ],
           [0.275   , 0.010547, 0.578254],
           [0.3     , 0.01875 , 0.45    ],
           [0.325   , 0.029297, 0.415314],
           [0.35    , 0.064931, 0.3125  ],
           [0.375   , 0.071057, 0.297703],
           [0.4     , 0.077182, 0.2634  ],
           [0.425   , 0.094922, 0.15559 ],
           [0.45    , 0.132312, 0.1125  ],
           [0.475   , 0.138438, 0.091884],
           [0.5     , 0.187443, 0.083308],
          [0.525   , 0.194793, 0.030628],
          [0.55    , 0.256049, 0.028178],
          [0.575   , 0.263672, 0.003125],
          [0.6     , 0.3     , 0.      ]])
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

        # Water-oil relative permeability SWT
        #      SW        krw       krow 
        self.SWT = np.array([
            (0.200000, 0.000000, 0.510200),
            (0.235714, 0.000010, 0.439917),
            (0.271429, 0.000163, 0.374841),
            (0.307143, 0.000824, 0.314970),
            (0.342857, 0.002603, 0.260306),
            (0.378571, 0.006355, 0.210848),
            (0.414286, 0.013177, 0.166596),
            (0.450000, 0.024412, 0.127550),
            (0.485714, 0.041647, 0.093710),
            (0.521429, 0.066710, 0.065077),
            (0.557143, 0.101676, 0.041649),
            (0.592857, 0.148864, 0.023428),
            (0.628571, 0.210836, 0.010412),
            (0.664286, 0.290398, 0.002603),
            (0.700000, 0.390600, 0.000000)], dtype=np.float64)

        self.krw = np.poly1d(np.polyfit(self.SWT[:, 0], self.SWT[:, 1], 2))
        self.kro = np.poly1d(np.polyfit(self.SWT[:, 0], self.SWT[:, 2], 2))


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
        



