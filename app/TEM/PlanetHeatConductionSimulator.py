import numpy as np

from fealpy.functionspace import ParametricLagrangeFiniteElementSpace

class PlanetHeatConductionSimulator():

    def __init__(self, mesh):
        self.space = ParametricLagrangeFiniteElementSpace(mesh, p=1)
        self.mesh = self.space.mesh
        
        self.M = self.space.mass_matrix()
        self.A = self.space.stiff_matrix()





if __name__ == '__main__':
    from fealpy.mesh import MeshFactory
    import matplotlib.pyplot as plt

    mesh = MeshFactory.unitcirclemesh(h=0.1, p=1)

    mesh.meshdata['A'] = 0.1 # 邦德反照率
    mesh.meshdata['epsilon'] = 0.9 # 辐射率
    mesh.meshdata['rho'] = 1400 # kg/m^3 密度
    mesh.meshdata['c'] = 1200 # Jkg^-1K^-1 比热容
    mesh.meshdata['kappa'] = 0.02 # Wm^-1K^-1 热导率
    mesh.meshdata['sigma'] = 5.6367e-8 # 玻尔兹曼常数
    mesh.meshdata['q'] = 1367.5 # W/m^2 太阳辐射通量
    mesh.meshdata['mu0'] =  1 # max(cos beta,0) 太阳高度角参数

    simulator  = PlanetHeatConductionSimulator(mesh)

    mesh.to_vtk(fname='test.vtu')






