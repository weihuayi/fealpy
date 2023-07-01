import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import TwoHolesData


class TwoHolesDataTest:
    def __init__(self):
        self.pde = TwoHolesData()

    def test_init_mesh(self):
        mesh = self.pde.init_mesh(h=0.05)
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        plt.show()


test = TwoHolesDataTest()
test.test_init_mesh()
