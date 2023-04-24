from types import ModuleType
from typing import Optional, Literal, Dict, Any, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.patches import Polygon
from matplotlib.axes import Axes
from .mesh import Mesh, Mesh1d, Mesh2d, Mesh3d


def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)

_MT = TypeVar('_MT', bound=Mesh)

class Plotable():
    """
    @brief A base class to make a mesh class plotable.
    """
    @property
    def add_plot(self):
        pass

    @property
    def find_node(self):
        pass

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
    def __init__(self, mesh: _MT, defaults: Dict[str, Any]) -> None:
        self._mesh = mesh
        self._defaults = defaults

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_axes(plot, projection: Optional[str]=None) -> Axes:
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')

            if projection in {'3', '3d', '3D'}:
                return fig.add_subplot(111, projection='3d')
            else:
                return fig.gca()
        else:
            return plot

    def set_show_axis(self, switch: bool):
        if switch:
            self._axes.set_axis_on()
        else:
            self._axes.set_axis_off()

    def set_aspect(self, aspect: Optional[float]=None):
        GD = self._mesh.geo_dimension()

        if (aspect is None) and (GD == 3):
            self._axes.set_box_aspect((1, 1, 1))
            self._axes.set_proj_type('ortho')

        elif (aspect is None) and (GD == 2):
            self._axes.set_box_aspect(1)

        else:
            self._axes.set_aspect(aspect=aspect)

    def set_lim(self, box: Optional[NDArray]=None):
        """
        @brief Set boundaries for plot area. By default, canvas can contain\
               the whole mesh.
        """
        GD = self._mesh.geo_dimension()
        if box is None:
            node: NDArray = self._mesh.entity('node').numpy()
            em: NDArray = self._mesh.entity_measure('edge').numpy()
            tol = np.max(em)/100
            box = np.zeros(2*GD, dtype=np.float64)
            box[0::2] = np.min(node, axis=0) - tol
            box[1::2] = np.max(node, axis=0) + tol

        self._axes.set_xlim(box[0:2])
        self._axes.set_ylim(box[2:4])

        if GD == 3:
            self._axes.set_zlim(box[4:6])

    def show_index(self):
        pass

    def point_scatter(self, points: Optional[NDArray], color, markersize):
        """
        @brief Show the node entities in the axes.

        @param points: An NDArray containing the node positions.
        @param color: Color of the node markers.
        @param markersize: The size of the node markers.

        @return: PathCollection.
        """
        if points is None:
            node: NDArray = self._mesh.entity('node').numpy()
        else:
            node = points

        GD = node.shape[-1]

        if GD == 2:
            return self._axes.scatter(
                node[:, 0], node[:, 1],
                color=color, s=markersize
            )
        elif GD == 3:
            return self._axes.scatter(
                node[:, 0], node[:, 1], node[:, 2],
                color=color, s=markersize
            )
        else:
            raise ValueError(f"Can only tackle the 'node' array with length\
                             2 or 3 in the last dimension.")

    def length_line(self, points: Optional[NDArray], struct: Optional[NDArray],
                    color, linewidths):
        """
        @brief Show the length entities as lines in the axes.

        @param points: An NDArray containing the node positions.
        @param struct: An NDArray of indices of nodes in two ends of an edge.
        @param color: Color of lines.
        @param linewidths: Widths of lines.

        @return: LineCollection or Line3DCollection.
        """
        if points is None:
            node: NDArray = self._mesh.entity('node').numpy()
        else:
            node = points

        if struct is None:
            cell: NDArray = self._mesh.entity('cell').numpy()
        else:
            cell = struct

        vts = node[cell, :]
        GD = node.shape[-1]

        if GD == 2:
            from matplotlib.collections import LineCollection
            lines = LineCollection(vts, linewidths=linewidths, colors=color)
            return self._axes.add_collection(lines)
        elif GD == 3:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lines = Line3DCollection(vts, linewidths=linewidths, colors=color)
            return self._axes.add_collection3d(vts)
        else:
            raise ValueError(f"Can only tackle the 'points' array with length\
                             2 or 3 in the last dimension.")

    def area_poly(self, points: Optional[NDArray], struct: Optional[NDArray],
                    edgecolor, cellcolor, linewidths):
        """
        @brief Show the area entities as polygons in the axes.

        @param points: An NDArray containing the node positions.
        @param struct: An NDArray of indices of nodes in every cells.

        @return: PolyCollection or Poly3DCollection.
        """
        if points is None:
            node: NDArray = self._mesh.entity('node').numpy()
        else:
            node = points

        if struct is None:
            cell: NDArray = self._mesh.entity('cell').numpy()
        else:
            cell = struct

        GD = node.shape[-1]

        if GD == 2:
            from matplotlib.collections import PolyCollection
            poly = PolyCollection(node[cell[:, self._mesh.ds.ccw], :])
        elif GD == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(node[cell[:, self._mesh.ds.ccw], :])
        else:
            raise ValueError(f"Can only tackle the 'points' array with length\
                             2 or 3 in the last dimension.")

        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(linewidths)
        poly.set_facecolor(cellcolor)
        return poly


class MeshCanvas1d(MeshCanvas[Mesh1d]):
    def __call__(self, plot: Axes,
            nodecolor='k', cellcolor='k',
            markersize=20, linewidths=1,
            shownode=True,
            aspect='equal', showaxis=False, box=None):
        self._axes = self.get_axes(plot)
        self.set_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        node: NDArray = self._mesh.entity('node').numpy()

        if node.ndim == 1:
            node = node[:, None]

        if node.shape[1] == 1:
            node = np.concatenate([node, np.zeros_like(node)], axis=-1)

        if shownode:
            self.point_scatter(points=node, color=nodecolor,
                               markersize=markersize)

        cell: NDArray = self._mesh.entity('cell').numpy()

        return self.length_line(points=node, struct=cell, color=cellcolor,
                                linewidths=linewidths)


class MeshCanvas2d(MeshCanvas[Mesh2d]):
    def __call__(
            self, plot: Axes,
            edgecolor='k', cellcolor=[0.5, 0.9, 0.45],
            aspect=None, linewidths: float=1.0,
            showaxis: bool=False, colorbar: bool=False,
            colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None
        ):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.

        Options accepted:
        """
        self._axes = self.get_axes(plot)
        self.set_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        if isinstance(cellcolor, Tensor):
            cellcolor = cellcolor.numpy()
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=cmap, cmax=cmax, cmin=cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if colorbar:
                f = self._axes.get_figure()
                f.colorbar(mapper, shrink=colorbarshrink, ax=self._axes)

        node: NDArray = self._mesh.entity('node').numpy()
        cell: NDArray = self._mesh.entity('cell').numpy()

        return self.area_poly(points=node, struct=cell,
                              edgecolor=edgecolor, cellcolor=cellcolor,
                              linewidths=linewidths) # TODO: Other types of meshes.


    def poly(self):
        """
        @brief Generate a PolyCollection object from the mesh.
        """
        mtype = getattr(self._mesh, 'meshtype', None)
        if mtype not in {'polygon', 'hepolygon', 'halfedge', 'halfedge2d'}:
            return self._poly_general()

    # PolyCollection for mesh with type not in 'polygon', 'hepolygon',
    # 'halfedge', and 'halfedge2d'.
    def _poly_general(self):
        node = self._mesh.entity('node')
        cell = self._mesh.entity('cell')
        GD = self._mesh.geo_dimension()

        if GD == 2:
            from matplotlib.collections import PolyCollection
            poly = PolyCollection(node[cell[:, self._mesh.ds.ccw], :])
        elif GD == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(node[cell[:, self._mesh.ds.ccw], :])

        return poly


class MeshCanvas3d(MeshCanvas[Mesh3d]):
    def __call__(
            self, plot: Axes,
            nodecolor='k', edgecolor='k', facecolor='w', cellcolor='w',
            aspect=[1, 1, 1],
            linewidths=0.5, markersize=20,
            showaxis=False, alpha=0.8, shownode=False, showedge=False, threshold=None
        ):
        self._axes = self.get_axes(plot, projection='3d')
        self.set_aspect(aspect)
        self.set_show_axis(showaxis)

        if (type(nodecolor) is np.ndarray) & np.isreal(nodecolor[0]):
            cmax = nodecolor.max()
            cmin = nodecolor.min()
            norm = colors.Normalize(vmin=cmin, vmax=cmax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node: NDArray = self._mesh.entity('node').numpy()
        if shownode:
            self._axes.scatter(
                    node[:, 0], node[:, 1], node[:, 2],
                    color=nodecolor, s=markersize)

        if showedge:
            edge: NDArray = self._mesh.entity('edge').numpy()
            vts = node[edge]
            edges = a3.art3d.Line3DCollection(
                   vts,
                   linewidths=linewidths,
                   color=edgecolor)
            return self._axes.add_collection3d(edges)

        face: NDArray = self._mesh.entity('face').numpy()
        isBdFace = self._mesh.ds.boundary_face_flag()
        if threshold is None:
            face = face[isBdFace][:, self._mesh.ds.ccw]
        else:
            bc = self._mesh.entity_barycenter('cell')
            isKeepCell = threshold(bc)
            face2cell = self._mesh.ds.face_to_cell()
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) & isBdFace
            face = face[isBdFace | isInterfaceFace][:, self._mesh.ds.ccw]

        faces = a3.art3d.Poly3DCollection(
                node[face],
                facecolor=facecolor,
                linewidths=linewidths,
                edgecolor=edgecolor,
                alpha=alpha)
        h = self._axes.add_collection3d(faces)
        box = np.zeros((2, 3), dtype=np.float_)
        box[0, :] = np.min(node, axis=0)
        box[1, :] = np.max(node, axis=0)
        self._axes.scatter(box[:, 0], box[:, 1], box[:, 2], s=0)
        return h


class EntityFind():
    def __init__(self) -> None:
        pass
