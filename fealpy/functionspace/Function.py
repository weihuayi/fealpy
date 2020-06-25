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

    def __getattr__(self, item):
        def wrap(func):
            def outer(*args,  **kwargs):
                val = func(self, *args, **kwargs)
                return val
            outer.coordtype = func.coordtype
            return outer 
        if hasattr(self.space, item):
            func = getattr(self.space, item)
            self.__dict__[item]= wrap(getattr(self.space, item))
            return self.__dict__[item]
        else:
            raise AttributeError()

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
        elif mesh.meshtype in {'polygon', 'halfedge', 'halfedge2d'}:
            node = mesh.entity('node')
            axes.plot_trisurf(
                    node[:, 0], node[:, 1], self, cmap=cmap, lw=0.0)
            return axes
        elif mesh.meshtype in {'stri'}:
            bc = np.array([1/3, 1/3, 1/3])
            mesh.add_plot(axes, cellcolor=self(bc), showcolorbar=True)
        else:
            return None
