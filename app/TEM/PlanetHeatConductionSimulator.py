import numpy as np

from fealpy.functionspace import ParametricLagrangeFiniteElementSpace

class PlanetHeatConductionSimulator():

    def __init__(self, mesh):
        self.space = ParametricLagrangeFiniteElementSpace(mesh, p=1)




if __name__ == '__main__':
    from fealpy.mesh import MeshFactory
    import matplotlib.pyplot as plt

    mesh = MeshFactory.unitcirclemesh(h=0.1, p=1)

    mesh.meshdata['A'] = 0.1 # 邦德反照率
    mesh.meshdata['epsilon'] = 0.9 # 辐射率
    mesh.meshdata['rho'] = 1400 # kg/m^3 密度

    simulator  = PlanetHeatConductionSimulator(mesh)

    mesh.to_vtk(fname='test.vtu')






