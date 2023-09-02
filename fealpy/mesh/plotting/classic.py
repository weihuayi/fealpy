
from typing import (
    Callable, Optional, TypeVar, Any, Union, Sequence, Generic,
    Dict, Type
)
from types import ModuleType
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from ..mesh_base import Mesh
from . import artist as A

_MT = TypeVar('_MT', bound=Mesh)

def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    from matplotlib import colors, cm

    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)

_ploter_map_: Dict[str, Type] = {}

def get_ploter(key: str) -> Type['MeshPloter']:
    if key in _ploter_map_:
        return _ploter_map_[key]
    else:
        raise KeyError(f"Can not find a ploter class that key '{key}' mapping to. "
                       "To use Plotable, register the target ploter first and then "
                       "specify the ploter for mesh by setting a same key. "
                       "See MeshPloter.register() and Plotable.set_ploter().")


class MeshPloter(Generic[_MT]):
    _axes: Axes
    _mesh: _MT

    def __init__(self, mesh: _MT) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("MeshPloter only works for mesh type, "
                            f"but got {self.__class__.__name__}.")
        self._mesh = mesh

    @property
    def current_axes(self):
        return self._axes

    @current_axes.setter
    def current_axes(self, axes: Axes):
        if isinstance(axes, Axes):
            self._axes = axes
        else:
            raise TypeError("Param 'axes' should be Axes type, "
                            f"but got {axes.__class__.__name__}.")

    @property
    def mesh(self):
        return self._mesh

    def _call_impl(self, axes: Union[Axes, ModuleType], *args, **kwargs):
        if isinstance(axes, ModuleType):
            fig = axes.figure()
            axes = fig.add_subplot(1, 1, 1)

        self.current_axes = axes

        return self.draw(*args, **kwargs)

    __call__ = _call_impl

    draw: Callable[..., Any]

    @classmethod
    def register(cls, key: str):
        """
        @brief Register this ploter with a unique string key.
        """
        if key in _ploter_map_:
            ploter = _ploter_map_[key]
            raise KeyError(f"Key '{key}' has been used by ploter {ploter.__name__}.")
        elif not issubclass(cls, MeshPloter):
            raise TypeError(f"Expect subclass of MeshPloter but got itself.")
        elif not isinstance(key, str):
            raise TypeError("Only accepts a single string as the key, "
                            f"but got {key.__class__.__name__}.")
        _ploter_map_[key] = cls

    def set_show_axis(self, switch: bool=True):
        if switch:
            self._axes.set_axis_on()
        else:
            self._axes.set_axis_off()

    def set_box_aspect(self, aspect: Optional[Union[float, Sequence[float]]]=None):
        if aspect is None:
            from mpl_toolkits.mplot3d import Axes3D

            if isinstance(self._axes, Axes3D):
                self._axes.set_box_aspect((1.0, 1.0, 1.0))
            else:
                self._axes.set_box_aspect(1.0)
        else:
            self._axes.set_box_aspect(aspect=aspect)

    def set_lim(self, box: Optional[NDArray]=None, tol=0.1):
        from mpl_toolkits.mplot3d import Axes3D

        GD = self._mesh.geo_dimension()

        if box is None:
            node: NDArray = self._mesh.entity('node')
            if node.ndim == 1:
                node = node.reshape(-1, 1)

            box = np.array([-0.5, 0.5]*3, dtype=np.float64)
            box[0:2*GD:2] = np.min(node, axis=0) - tol
            box[1:1+2*GD:2] = np.max(node, axis=0) + tol

        self._axes.set_xlim(box[0:2])
        self._axes.set_ylim(box[2:4])

        if isinstance(self._axes, Axes3D):
            self._axes.set_zlim(box[4:6])


class EntityFinder(MeshPloter):
    def draw(self, etype_or_node: Union[int, str, NDArray], index=np.s_[:],
                showindex: bool=False, multiindex: Optional[Sequence[Any]]=None,
                color='r', marker='o', markersize=20,
                fontcolor='k', fontsize=24):
        """
        @brief Show the barycenter of each entity.
        """
        from ..plotting import artist as A
        from ..plotting.classic import array_color_map

        axes = self.current_axes

        if isinstance(etype_or_node, (int, str)):
            bc = self.mesh.entity_barycenter(etype=etype_or_node, index=index)
        elif isinstance(etype_or_node, np.ndarray):
            bc = etype_or_node
        else:
            raise TypeError(f"Invalid entity type or node info.")
        if bc.ndim == 1:
            bc = bc[:, None]

        if isinstance(color, np.ndarray) and np.isreal(color[0]):
            mapper = array_color_map(color, 'rainbow')
            color = mapper.to_rgba(color)

        A.scatter(axes=axes, points=bc, color=color,
                  marker=marker, markersize=markersize)
        if showindex:
            if multiindex is None:
                A.show_index(axes=axes, location=bc, number=index,
                            fontcolor=fontcolor, fontsize=fontsize)
            else:
                A.show_multi_index(axes=axes, location=bc, text_list=multiindex,
                                   fontcolor=fontcolor, fontsize=fontsize)

EntityFinder.register('finder')


##################################################
### MeshPloter subclasses
##################################################


class AddPlot1d(MeshPloter):
    def draw(
            self, nodecolor='k', cellcolor='k',
            markersize=20, linewidths=1,
            aspect=None,
            shownode=True, showaxis=False,
            box=None, **kwargs
        ):
        axes = self.current_axes
        self.set_box_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        node: NDArray = self.mesh.entity('node')

        if node.ndim == 1:
            node = node[:, None]

        if shownode:
            A.scatter(axes=axes, points=node, color=nodecolor,
                      markersize=markersize)

        cell = self.mesh.entity('cell')

        return A.line(axes=axes, points=node, struct=cell, color=cellcolor,
                      linewidths=linewidths)

AddPlot1d.register('1d')


class AddPlot2dHomo(MeshPloter):
    def draw(
            self, edgecolor='k', cellcolor='#99BBF6',
            linewidths: float=1.0, alpha: float=1.0,
            aspect=None,
            showaxis: bool=False, colorbar: bool=False, colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None, **kwargs
        ):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.
        """
        axes = self.current_axes
        self.set_box_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=cmap, cmax=cmax, cmin=cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if colorbar:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=colorbarshrink, ax=self._axes)

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        ccw = getattr(self.mesh.ds, 'ccw', None)
        if ccw is not None:
            cell = cell[..., ccw]

        return A.poly(axes=axes, points=node, struct=cell,
                      edgecolor=edgecolor, cellcolor=cellcolor,
                      linewidths=linewidths, alpha=alpha)

AddPlot2dHomo.register('2d')


class AddPlot2dPoly(MeshPloter):
    def draw(
            self, edgecolor='k', cellcolor='#99BBF6',
            linewidths: float=1.0, alpha: float=1.0,
            aspect=None,
            showaxis: bool=False, colorbar: bool=False, colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None, **kwargs
        ):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.
        """
        axes = self.current_axes
        self.set_box_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=cmap, cmax=cmax, cmin=cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if colorbar:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=colorbarshrink, ax=self._axes)

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell') # A list containing indices of vertices.

        return A.poly_(axes=axes, points=node, struct_seq=cell,
                       edgecolor=edgecolor, cellcolor=cellcolor,
                      linewidths=linewidths, alpha=alpha)

AddPlot2dPoly.register('polygon2d')


class AddPlot3dHomo(MeshPloter):
    def draw(
            self, nodecolor='k', edgecolor='k', cellcolor='w',
            markersize=20, linewidths=0.5, alpha=0.8,
            aspect: Optional[Union[float, Sequence[float]]]=None,
            showaxis=False, shownode=False, showedge=False, threshold=None,
            box=None, **kwargs
        ):
        axes = self.current_axes
        self.set_box_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        if isinstance(nodecolor, np.ndarray) and np.isreal(nodecolor[0]):
            mapper = array_color_map(nodecolor, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node = self.mesh.entity('node')
        if shownode:
            A.scatter(axes=axes, points=node, color=nodecolor, markersize=markersize)

        if showedge:
            edge = self.mesh.entity('edge')
            A.line(axes=axes, points=node, struct=edge, color=edgecolor,
                   linewidths=linewidths)

        face = self.mesh.entity('face')
        ccw = getattr(self.mesh.ds, 'ccw', None)
        if ccw is not None:
            face = face[..., ccw]
        isBdFace = self.mesh.ds.boundary_face_flag()

        if threshold is None:
            face = face[isBdFace]
        else:
            bc = self.mesh.entity_barycenter('cell')
            isKeepCell = threshold(bc)
            face2cell = self.mesh.ds.face_to_cell()
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) and isBdFace
            face = face[isBdFace | isInterfaceFace]

        return A.poly(axes=axes, points=node, struct=face, edgecolor=edgecolor,
                      cellcolor=cellcolor, linewidths=linewidths, alpha=alpha)

AddPlot3dHomo.register('3d')
