
from typing import Optional, Union, Dict, Any, TypeVar, Generic

from numpy.typing import NDArray
from matplotlib.collections import (
    PathCollection, LineCollection, PolyCollection, Collection
)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.axes import Axes
from .mesh import Mesh, Mesh1d, Mesh2d, Mesh3d


_MT = TypeVar('_MT', bound=Mesh)

class Plotable():
    """
    @brief A base class to make a mesh class plotable.
    """
    def add_plot(
            self, plot: Axes,
            nodecolor=..., edgecolor=..., facecolor=..., cellcolor=...,
            markersize: float=..., linewidths: float=..., alpha: float=...,
            aspect=...,
            showaxis: bool=..., shownode: bool=..., showedge: bool=...,
            colorbar: bool=..., colorbarshrink: float=...,
            cmax: float=..., cmin: float=..., cmap=..., box=...
        ) -> Collection: ...

    def find_node(self, axes: Axes, node=None,
            index: slice=...,
            showindex: bool=False,
            color='r', markersize=20,
            fontsize=16, fontcolor='r',
            multiindex=None): ...

    @property
    def find_edge(self):
        pass

    @property
    def find_face(self):
        pass

    @property
    def find_cell(self):
        pass


class MeshCanvas(Generic[_MT]):
    _axes: Axes
    def __init__(self, mesh: _MT, defaults: Dict[str, Any]) -> None: ...

    def __call__(self, *args: Any, **kwds: Any) -> Any: ...

    @staticmethod
    def get_axes(plot, projection: Optional[str]=None) -> Axes: ...
    def set_show_axis(self, switch: bool) -> None: ...
    def set_aspect(self, aspect: Optional[float]=None) -> None: ...
    def set_lim(self, box: Optional[NDArray]=None) -> None: ...
    def show_index(self) -> None: ...
    def point_scatter(self, points: Optional[NDArray], color, markersize) -> PathCollection: ...
    def length_line(self, points: Optional[NDArray], struct: Optional[NDArray],
                    color, linewidths) -> Union[LineCollection, Line3DCollection]: ...
    def area_poly(
            self, points: Optional[NDArray], struct: Optional[NDArray],
            edgecolor, cellcolor, linewidths
            ) -> Union[PolyCollection, Poly3DCollection]: ...


class MeshCanvas1d(MeshCanvas[Mesh1d]):
    def __call__(
            self, plot: Axes,
            nodecolor='k', cellcolor='k',
            markersize=20, linewidths=1,
            aspect='equal',
            shownode=True, showaxis=False,
            box=None
        ) -> Union[LineCollection, Line3DCollection]: ...


class MeshCanvas2d(MeshCanvas[Mesh2d]):
    def __call__(
            self, plot: Axes,
            edgecolor='k', cellcolor=[0.5, 0.9, 0.45],
            linewidths: float=1.0,
            aspect=None,
            showaxis: bool=False, colorbar: bool=False, colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None
        ) -> Union[PolyCollection, Poly3DCollection]: ...


class MeshCanvas3d(MeshCanvas[Mesh3d]):
    def __call__(
            self, plot: Axes,
            nodecolor='k', edgecolor='k', cellcolor='w',
            markersize=20, linewidths=0.5, alpha=0.8,
            aspect=[1, 1, 1],
            showaxis=False, shownode=False, showedge=False, threshold=None,
            box=None
        ) -> Poly3DCollection: ...


class EntityFind():
    def __init__(self) -> None:
        pass
