
from typing import Callable, Optional, TypeVar, Any, Union, Sequence, Generic
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh.mesh_base import Mesh
from ..mesh_base import Mesh, Mesh1d, Mesh2d, Mesh3d
from . import artist as A

_MT = TypeVar('_MT', bound=Mesh)

def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


class Plotable():
    def __init__(self, mesh) -> None:
        self._mesh = mesh

    @property
    def add_plot(self):
        return self._get_ploter()

    def _get_ploter(self):
        pass


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
        return self.mesh

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
            if isinstance(self._axes, Axes3D):
                self._axes.set_box_aspect((1.0, 1.0, 1.0))
            else:
                self._axes.set_box_aspect(1.0)
        else:
            self._axes.set_box_aspect(aspect=aspect)

    def set_lim(self, box: Optional[NDArray]=None):
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

    def find_cell(self):
        pass


class AddPlot1d(MeshPloter[Mesh1d]):
    def draw(self):
        pass


class AddPlot2d(MeshPloter[Mesh2d]):
    def draw(self):
        pass


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

        node: NDArray = self._mesh.entity('node')
        if shownode:
            A.scatter(axes=axes, points=node, color=nodecolor, markersize=markersize)

        if showedge:
            edge: NDArray = self._mesh.entity('edge')
            A.line(axes=axes, points=node, struct=edge, color=edgecolor,
                   linewidths=linewidths)

        face: NDArray = self._mesh.entity('face')
        ccw = getattr(self._mesh.ds, 'ccw', None)
        if ccw is not None:
            face = face[..., ccw]
        isBdFace = self._mesh.ds.boundary_face_flag()

        if threshold is None:
            face = face[isBdFace]
        else:
            bc = self._mesh.entity_barycenter('cell')
            isKeepCell = threshold(bc)
            face2cell = self._mesh.ds.face_to_cell()
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) and isBdFace
            face = face[isBdFace | isInterfaceFace]

        poly = A.poly(axes=axes, points=node, struct=face, edgecolor=edgecolor,
                      cellcolor=cellcolor, linewidths=linewidths, alpha=alpha)
        return axes.add_collection(poly)


_DEFAULTS = {
    'nodecolor': 'r',
    'edgecolor': 'k',
    'facecolor': 'cy',
    'cellcolor': [0.2, 0.6, 1.0],

    'markersize': 10.0,
    'linewidths': 0.1,
    'alpha': 1.0,
}


class PlotModule():
    def __init__(self, **kwargs) -> None:
        self._options = kwargs

    def _get_options(self, key: str):
        if key in self._options:
            return self._options[key]
        elif key in _DEFAULTS:
            return _DEFAULTS[key]
        else:
            raise KeyError(f"Failed to load default value of {key}.")

    def _call_impl(self, axes: Axes, mesh: Mesh):
        return self.draw(axes=axes, mesh=mesh)

    __call__ = _call_impl

    def draw(self, axes: Axes, mesh: Mesh):
        raise NotImplementedError
