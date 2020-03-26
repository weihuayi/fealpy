import numpy as np
from types import ModuleType


class Function(np.ndarray):
    def __new__(cls, space, dim=None, array=None):
        if array is None:
            self = space.array(dim=dim).view(cls)
        else:
            self = array.view(cls)
        self.space = space
        return self

    def index(self, i):
        return Function(self.space, array=self[:, i])

    def __call__(self, bc, index=None):
        space = self.space
        return space.value(self, bc, index=index)

    def value(self, bc, index=None):
        space = self.space
        return space.value(self, bc, index=index)

    def grad_value(self, bc, index=None):
        space = self.space
        return space.grad_value(self, bc, index=index)

    def laplace_value(self, bc, index=None):
        space = self.space
        return space.laplace_value(self, bc, index=index)

    def div_value(self, bc, index=None):
        space = self.space
        return space.div_value(self, bc, index=index)

    def hessian_value(self, bc, index=None):
        space = self.space
        return space.hessian_value(self, bc, index=index)

    def edge_value(self, bc, index=None):
        space = self.space
        return space.edge_value(self, bc)

    def add_plot(self, plot, cmap=None):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        mesh = self.space.mesh
        if mesh.meshtype == 'tri':
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            axes.plot_trisurf(
                    node[:, 0], node[:, 1],
                    cell, self, cmap=cmap, lw=0.0)
            return axes
        elif mesh.meshtype in {'polygon', 'halfedge'}:
            node = mesh.entity('node')
            axes.plot_trisurf(
                    node[:, 0], node[:, 1], self, cmap=cmap, lw=0.0)
            return axes
        else:
            return None
