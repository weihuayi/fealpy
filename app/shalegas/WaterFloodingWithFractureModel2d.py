
import numpy as np

class WaterFloodingWithFractureModel2d():
    """

    Notes
    -----

    这是一个二维带裂缝的水驱模型

    模型单位采用标准国际单元：

    * 时间： s
    * 长度： m
    * 压强： Pa
    """

    def __init__(self):
        self.domain=[0, 10, 0, 10] # m
        self.rock = {
            'permeability': 2, # 1 d = 9.869 233e-13 m^2 
            'porosity': 0.3, # None
            'lame':(1.0e+2, 3.0e+2), # lambda and mu 拉梅常数, MPa
            'biot': 1.0,
            'initial pressure': 3, # MPa
            'initial stress': 2.0e+2, # MPa 初始应力 sigma_0 , sigma_eff
            'solid grain stiffness': 2.0e+2, # MPa 固体体积模量
            }

        self.fracture = {
            'permeability': 4, # 1 d = 9.869 233e-13 m^2 
            'porosity': 0.3, # None
            'point': np.array([
                (1, 1),
                (9, 9),
                (9, 1),
                (1, 9)], dtype=np.float64),
            'segment': np.array([(0, 1), (2, 3)], dtype=np.int_)
                }

        self.water = {
            'viscosity': 1, # 1 cp = 1 mPa*s
            'compressibility': 1.0e-3, # MPa^{-1}
            'initial saturation': 0.0, 
            'injection rate': 3.51e-6 # s^{-1}, 每秒注入多少水
            }

        self.oil = {
            'viscosity': 2, # cp
            'compressibility': 2.0e-3, # MPa^{-1}
            'initial saturation': 1.0, 
            'production rate': 3.50e-6 # s^{-1}, 每秒产出多少油
            }

        self.bc = {'displacement': 0.0, 'flux': 0.0}

        self.GD = 2

    def space_mesh(self, n=10):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=1, ny=1, meshtype='tri')
        point = self.fracture['point']
        segment = self.fracture['segment']
        for i in range(n):
            isCutCell= mesh.find_segment_location(point, segment)
            mesh.bisect(isCutCell)
        return mesh

    def is_fracture_cell(self, mesh):
        point = self.fracture['point']
        segment = self.fracture['segment']
        isFCell= mesh.find_segment_location(point, segment)
        return isFCell

    def time_mesh(self, T0=0, T1=1, n=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(T0, T1, n)
        return timeline

    def water_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算水的相对渗透率
        """
        val = Sw**2
        return val

    def oil_relative_permeability(self, Sw):
        """

        Notes
        ----
        给定水的饱和度, 计算油的相对渗透率
        """
        val = (1 - Sw)**2 
        return val
