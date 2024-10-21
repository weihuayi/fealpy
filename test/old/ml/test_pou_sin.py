import torch
from fealpy.mesh import UniformMesh2d
from fealpy.ml.modules import PoUSin, Standardize, TensorMapping
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


mesh = UniformMesh2d((0, 4, 0, 4), h=(0.25, 0.25), origin=(0.0, 0.0))
node = torch.from_numpy(mesh.entity('node'))

class TheTestModel(TensorMapping):
    def __init__(self):
        super().__init__()
        self.l1 = Standardize(centers=node, radius=0.125)
        self.l2 = PoUSin()

    def forward(self, p):
        ret = self.l1(p)
        ret = self.l2(ret)
        weight = torch.arange(25, dtype=torch.float64)/2
        return torch.einsum('nm, m -> n', ret, weight).unsqueeze(-1)


if __name__ == '__main__':

    model = TheTestModel()

    x = np.linspace(0, 1, 190)
    y = np.linspace(0, 1, 190)
    data, (mx, my) = model.meshgrid_mapping(x, y)

    fig = plt.figure("Test for PoU")
    axes = fig.add_subplot(111, projection='3d')
    axes.plot_surface(mx, my, data, cmap=cm.RdYlBu_r, edgecolor='blue',
                    linewidth=0.0003, antialiased=True)
    plt.show()
