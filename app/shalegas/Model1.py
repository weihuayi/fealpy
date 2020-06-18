
import numpy as np
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.timeintegratoralg.timeline import UniformTimeLine

class Model1():
    def __init__(self):
        self.domain = [0, 50, 0, 50]
        self.mesh = MeshFactory().regular(self.domain, n=50)
        self.timeline = UniformTimeLine(0, 1, 100) 
        self.space0 = RaviartThomasFiniteElementSpace2d(self.mesh, p=0)
        self.space1 = ScaledMonomialSpace2d(self.mesh, p=1) # 线性间断有限元空间

        self.vh = self.space0.function() # 速度
        self.ph = self.space0.smspace.function() # 压力
        self.ch = self.space1.function(dim=3) # 三个组分的摩尔密度
        self.options = {
                'viscosity': 1.0, 
                'permeability': 1.0,
                'temperature': 397,
                'pressure': 50,
                'porosity': 0.2,
                'injecttion_rate': 0.1,
                'compressibility': (0.001, 0.001, 0.001),
                'pmv': (1.0, 1.0, 1.0)}

        c = self.options['viscosity']/self.options['permeability']
        self.A = c*self.space0.mass_matrix()
        self.B = self.space0.div_matrix()
        self.M = self.space0.smspace.mass_matrix() #  

    def get_current_left_matrix(self, data, timeline):
        pass

    def get_current_right_vector(self, data, timeline):
        pass

    def solve(self, data, A, b, solver, timeline):
        pass




if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model1()

    NN = model.mesh.number_of_nodes()
    print(NN)
    fig = plt.figure()
    axes = fig.gca()
    model.mesh.add_plot(axes)
    plt.show()
