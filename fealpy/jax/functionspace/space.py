
from typing import Union, Callable, Optional, Generic, TypeVar
from abc import ABCMeta, abstractmethod

import jax.numpy as jnp

from ..mesh.utils import Array

Index = Union[int, slice, Array]
Number = Union[int, float]
_S = slice(None)


class _FunctionSpace(metaclass=ABCMeta):
    r"""THe base class of function spaces"""
    ftype: jnp.dtype
    itype: jnp.dtype
    doforder: str='vdims'

    # basis
    @abstractmethod
    def basis(self, p: Array, index: Index=_S, **kwargs) -> Array: raise NotImplementedError
    @abstractmethod
    def grad_basis(self, p: Array, index: Index=_S, **kwargs) -> Array: raise NotImplementedError
    @abstractmethod
    def hess_basis(self, p: Array, index: Index=_S, **kwargs) -> Array: raise NotImplementedError

    # values
    @abstractmethod
    def value(self, uh: Array, p: Array, index: Index=_S) -> Array: raise NotImplementedError
    @abstractmethod
    def grad_value(self, uh: Array, p: Array, index: Index=_S) -> Array: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> Array: raise NotImplementedError
    def face_to_dof(self) -> Array: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., Array], Array, Number],
                    uh: Array, dim: Optional[int]=None, index: Index=_S) -> Array:
        raise NotImplementedError

    # function
    def array(self, dim: int=0) -> Array:
        GDOF = self.number_of_global_dofs()
        kwargs = {'dtype': self.ftype}

        if dim  == 0:
            shape = (GDOF, )
        else:
            shape = (GDOF, dim)

        return jnp.zeros(shape, **kwargs)


class Function(Array):
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
    def __init__(self, space, dim=None, array=None, coordtype=None,
            dtype=jnp.float64):
        if array is None:
            self.array = space.array(dim=dim, dtype=dtype)
        else:
            self.array = array
        self.space = space
        self.coordtype = coordtype
        # return self

    def index(self, i):
        return Function(self.space, array=self[:, i], coordtype=self.coordtype)

    def __call__(self, bc, index=jnp.s_[:]):
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
                val = jnp.repeat(self(bc), NV)
                cell, cellLocation = mesh.entity('cell')
                uh = jnp.zeros(NN, dtype=mesh.ftype)
                deg = jnp.zeros(NN, dtype=mesh.itype)
                jnp.add.at(uh, cell, val)
                jnp.add.at(deg, cell, 1)
                uh /= deg
                axes.plot_trisurf(
                        node[:, 0], node[:, 1], uh, cmap=cmap, lw=0.0)
            else:
                axes.plot_trisurf(
                        node[:, 0], node[:, 1], self, cmap=cmap, lw=0.0)
            return axes
        elif mesh.meshtype in {'stri'}:
            bc = jnp.array([1/3, 1/3, 1/3])
            mesh.add_plot(axes, cellcolor=self(bc), showcolorbar=True)
        else:
            return None
    
    
class FunctionSpace(_FunctionSpace):
    def function(self, dim=None, array=None, dtype=jnp.float64):
        return Function(self, dim=dim, array=array,
                coordtype='barycentric', dtype=dtype).array
    
    def array(self, dim=None, dtype=jnp.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            dim = tuple()
        if type(dim) is int:
            dim = (dim, )

        if self.doforder == 'sdofs':
            shape = dim + (gdof, )
        elif self.doforder == 'vdims':
            shape = (gdof, ) + dim
        return jnp.zeros(shape, dtype=dtype)
