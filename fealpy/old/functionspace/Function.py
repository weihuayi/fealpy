import numpy as np
from types import ModuleType

class Function(np.ndarray):
    """

    Notes
    -----
    Function 代表离散空间 space 中的函数, 同时它也是一个一维或二维数组, 形状通常为
    (gdof, ...), 其中 gdof 代表离散空间的维数, 第 1 个轴是变量的维数. 

    Examples
    --------
    >> import numpy as np
    >> from fealpy.pde.poisson_2d import CosCosData
    >> from fealpy.functionspace import 
    """
    def __new__(cls, space, dim=None, array=None, coordtype=None,
            dtype=np.float64):
        if array is None:
            self = space.array(dim=dim, dtype=dtype).view(cls)
        else:
            self = array.view(cls)
        self.space = space
        self.coordtype = coordtype
        return self

    def index(self, i):
        return Function(self.space, array=self[:, i], coordtype=self.coordtype)

    def __call__(self, bc, index=np.s_[:]):
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
            self.__dict__[item]= wrap(getattr(self.space, item))
            return self.__dict__[item]
        else:
            return self.__dict__[item]
        
    def add_plot(self, plot, cmap=None, threshold=None):
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = plot.axes(projection='3d')
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
        elif mesh.meshtype == 'tet': #TODO: make it work!
            space = self.space
            face = mesh.boundary_face(threshold=threshold) 
            node = mesh.entity('node')
            axes.plot_trisurf(node[:, 0], node[:, 1], node[:, 2],
                    triangles=face, cmap=cmap)
        elif mesh.meshtype in {'polygon', 'halfedge', 'halfedge2d'}:
            node = mesh.entity('node')
            if self.space.stype == 'wg':
                NN = mesh.number_of_nodes()
                NV = mesh.number_of_vertices_of_cells()
                bc = mesh.entity_barycenter('cell')
                val = np.repeat(self(bc), NV)
                cell, cellLocation = mesh.entity('cell')
                uh = np.zeros(NN, dtype=mesh.ftype)
                deg = np.zeros(NN, dtype=mesh.itype)
                np.add.at(uh, cell, val)
                np.add.at(deg, cell, 1)
                uh /= deg
                axes.plot_trisurf(
                        node[:, 0], node[:, 1], uh, cmap=cmap, lw=0.0)
            else:
                axes.plot_trisurf(
                        node[:, 0], node[:, 1], self, cmap=cmap, lw=0.0)
            return axes
        elif mesh.meshtype in {'stri'}:
            bc = np.array([1/3, 1/3, 1/3])
            mesh.add_plot(axes, cellcolor=self(bc), showcolorbar=True)
        else:
            return None
