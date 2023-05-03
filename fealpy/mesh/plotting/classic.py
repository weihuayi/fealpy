
from typing import (
    Callable, Optional, TypeVar, Any, Union, Sequence, Generic,
    Type
)
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from ..mesh_base import Mesh, Mesh1d, Mesh2d, Mesh3d
from . import artist as A

_MT = TypeVar('_MT', bound=Mesh)

def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    from matplotlib import colors, cm

    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


class MeshPloter(Generic[_MT]):
    _axes: Axes
    _mesh: _MT

    def __init__(self, mesh: _MT) -> None:
        self._mesh = mesh

    @property
    def current_axes(self):
        return self._axes

    @current_axes.setter
    def current_axes(self, axes: Axes):
        if isinstance(axes, Axes):
            self._axes = axes
        else:
            raise TypeError("Param 'axes' should be Axes type"
                            f"but got {axes.__class__.__name__}.")

    @property
    def mesh(self):
        return self._mesh

    def _call_impl(self, axes: Axes, *args, **kwargs):
        self.current_axes = axes

        return self.draw(*args, **kwargs)

    __call__ = _call_impl

    draw: Callable[..., Any]

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

    def set_lim(self, box: Optional[NDArray]=None):
        from mpl_toolkits.mplot3d import Axes3D

        GD = self._mesh.geo_dimension()
        if box is None:
            node: NDArray = self._mesh.entity('node')
            em: NDArray = self._mesh.entity_measure('edge')
            tol = np.max(em)/100
            box = np.zeros(2*GD, dtype=np.float64)
            box[0::2] = np.min(node, axis=0) - tol
            box[1::2] = np.max(node, axis=0) + tol

        self._axes.set_xlim(box[0:2])
        self._axes.set_ylim(box[2:4])

        if isinstance(self._axes, Axes3D):
            self._axes.set_zlim(box[4:6])


class Plotable():
    """
    @brief Base class for plotable meshes. Inherit this class to obtain several\
           plotting methods.

    Before using these plotting methods, call the class method
    `MeshClass.set_ploter()` to chose a proper ploter for the mesh type.
    For example, specity a plotter for a mesh type named 'TriangleMesh':
    ```
        class TriangleMesh(Mesh2d, Plotable):
            # codes for mesh ...
            ...

        TriangleMesh.set_ploter(MyPloter)
    ```
    Here `MyPloter` is a subclass of `MeshPloter`. Then, instances of `TriangleMesh`
    may call methods such as `add_plot()` to draw the mesh.

    @seealso: MeshPloter.
    """
    _ploter_class: Optional[Type[MeshPloter]] = None

    @property
    def add_plot(self):
        if self._ploter_class is not None:
            return self._ploter_class(self)
        else:
            raise Exception('MeshPloter of the type of mesh should be specified'
                            'before drawing. If a mesh is inherited from Plotable,'
                            'use MeshClass.set_ploter(MeshPloterClass) to specify.')

    @classmethod
    def set_ploter(cls, ploter: Type[MeshPloter]):
        cls._ploter_class = ploter

    def find_entity(self, axes: Axes, etype: Union[int, str], index=np.s_[:],
                    showindex: bool=False, color='r', markersize=20,
                    fontcolor='k', fontsize=24):
        """
        @brief Show barycenters of the entity.
        """
        if not isinstance(self, Mesh):
            raise TypeError("Plotable only works for mesh type,"
                            f"but got {self.__class__.__name__}.")

        bc = self.entity_barycenter(etype=etype, index=index)
        if bc.ndim == 1:
            bc = bc[:, None]

        if isinstance(color, np.ndarray) and np.isreal(color[0]):
            mapper = array_color_map(color, 'rainbow')
            color = mapper.to_rgba(color)

        A.scatter(axes=axes, points=bc, color=color, markersize=markersize)
        if showindex:
            if index == np.s_[:]:
                index = np.arange(bc.shape[0])
            elif isinstance(index, np.ndarray):
                if (index.dtype is np.bool_):
                    index, = np.nonzero(index)
            else:
                raise TypeError("Unknown index format.")
            A.show_index(axes=axes, location=bc, number=index,
                         fontcolor=fontcolor, fontsize=fontsize)

    def find_node(self, axes,
            index=np.s_[:],
            showindex=False,
            color='r', markersize=20,
            fontsize=16, fontcolor='r',
            multi_index=None):
        return self.find_entity(
                axes, 'node', index=index,
                showindex=showindex,
                color=color,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_edge(self, axes,
            index=np.s_[:],
            showindex=False,
            color='g', markersize=22,
            fontsize=18, fontcolor='g'):
        return self.find_entity(
                axes, 'edge', index=index,
                showindex=showindex,
                color=color,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_face(self, axes,
            index=np.s_[:],
            showindex=False,
            color='b', markersize=24,
            fontsize=20, fontcolor='b'):
        return self.find_entity(
                axes, 'face', index=index,
                showindex=showindex,
                color=color,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_cell(self, axes,
            index=np.s_[:],
            showindex=False,
            color='y', markersize=26,
            fontsize=22, fontcolor='y'):
        return self.find_entity(
                axes, 'cell', index=index,
                showindex=showindex,
                color=color,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)


##################################################
### MeshPloter subclasses
##################################################


class AddPlot1d(MeshPloter[Mesh1d]):
    def draw(
            self, nodecolor='k', cellcolor='k',
            markersize=20, linewidths=1,
            aspect='equal',
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


class AddPlot2dHomo(MeshPloter[Mesh2d]):
    def draw(
            self, edgecolor='k', cellcolor=[0.5, 0.9, 0.45],
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


class AddPlot2dPoly(MeshPloter[Mesh2d]):
    def draw(
            self, edgecolor='k', cellcolor=[0.5, 0.9, 0.45],
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


class AddPlot3d(MeshPloter[Mesh3d]):
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


# _DEFAULTS = {
#     'nodecolor': 'r',
#     'edgecolor': 'k',
#     'facecolor': 'cy',
#     'cellcolor': [0.2, 0.6, 1.0],

#     'markersize': 10.0,
#     'linewidths': 0.1,
#     'alpha': 1.0,
# }


# class PlotModule():
#     def __init__(self, **kwargs) -> None:
#         self._options = kwargs

#     def _get_options(self, key: str):
#         if key in self._options:
#             return self._options[key]
#         elif key in _DEFAULTS:
#             return _DEFAULTS[key]
#         else:
#             raise KeyError(f"Failed to load default value of {key}.")

#     def _call_impl(self, axes: Axes, mesh: Mesh):
#         return self.draw(axes=axes, mesh=mesh)

#     __call__ = _call_impl

#     def draw(self, axes: Axes, mesh: Mesh):
#         raise NotImplementedError
