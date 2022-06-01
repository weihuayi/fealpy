
"""

Notes
-----

Function 是有限维空间的函数类
"""
import numpy as np

class Function(object):
    MAGIC_METHODS = ('__radd__',
                     '__add__',
                     '__sub__',
                     '__rsub__',
                     '__mul__',
                     '__rmul__',
                     '__div__',
                     '__rdiv__',
                     '__pow__',
                     '__rpow__',
                     '__eq__',
                     '__len__',
                     'copy')

    class __metaclass__(type):
        def __init__(cls, name, parents, attrs):
            def make_delegate(name):
                def delegate(self, *args, **kwargs):
                    return getattr(self.data, name)
                return delegate
            type.__init__(cls, name, parents, attrs)
            for method_name in cls.MAGIC_METHODS:
                setattr(cls, method_name, property(make_delegate(method_name)))

    def __init__(self, space, dim=None, array=None):
        self.space = space
        self.coordtype = space.value.coordtype # 空间函数的坐标类型
        if array is None:
            self.data = space.array(dim=dim)
        else:
            self.data = array

    def index(self, i):
        assert len(self.data.shape) > 1
        return Function(self.space, array=self.data[:, i])

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __call__(self, bc, index=np.s_[:]):
        space = self.space
        return space.value(self.data, bc, index=index)

    def __getattr__(self, item):
        def wrap(func):
            def outer(*args,  **kwargs):
                val = func(self.data, *args, **kwargs)
                return val
            outer.coordtype = func.coordtype
            return outer 
        if hasattr(self.space, item):
            self.__dict__[item]= wrap(getattr(self.space, item))
            return self.__dict__[item]
        elif hasattr(self.data, item):
            self.__dict__[item] = getattr(self.data, item)
            return self.__dict__[item]
        else:
            return self.__dict__[item]

    def add_plot(self, plot, cmap=None, threshold=None):
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot

        mesh = self.space.mesh
        if mesh.meshtype == 'tri':
            space = self.space
            ipoints = space.interpolation_points()
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            axes.plot_trisurf(
                    ipoints[:, 0], ipoints[:, 1],
                    self, cmap=cmap, lw=0.0)
            return axes
        elif mesh.meshtype == 'tet':#TODO: make it work!
            space = self.space
            face = mesh.boundary_face(threshold=threshold) 
            node = mesh.entity('node')
            axes.plot_trisurf(node[:, 0], node[:, 1], node[:, 2],
                    triangles=face, cmap=cmap)
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
