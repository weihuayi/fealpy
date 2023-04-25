from types import ModuleType
from typing import Optional, Any, TypeVar, Generic, Union

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


class Plotable():
    """
    @brief A base class to make a mesh class plotable.
    """
    _ploter: Union['MeshCanvas1d', 'MeshCanvas2d', 'MeshCanvas3d']

    def _get_ploter(self, axes: Optional[Axes]=None):
        ploter = getattr(self, '_ploter', None)
        if (ploter is None) or (axes is not self._ploter._axes):
            if isinstance(self, Mesh1d):
                self._ploter = MeshCanvas1d(self)
            elif isinstance(self, Mesh2d):
                self._ploter = MeshCanvas2d(self)
            elif isinstance(self, Mesh3d):
                self._ploter = MeshCanvas3d(self)
            else:
                raise TypeError(f"Unsupported mesh type: {self.__class__.__name__}.")

        return self._ploter

    @property
    def add_plot(self):
        return self._get_ploter()

    def find_node(self, axes,
            index=np.s_[:],
            showindex=False,
            color='r', markersize=20,
            fontsize=16, fontcolor='r',
            multi_index=None):
        if multi_index is None:
            return self.find_entity(
                    axes, 'node', index=index,
                    showindex=showindex,
                    color=color,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)
        else:
            bc = self.find_entity(
                    axes, 'node', index=index,
                    showindex=False,
                    color=color,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)
            ploter = self._get_ploter(axes)
            ploter.show_multi_index(bc, multi_index=multi_index,
                                    fontcolor=fontcolor, fontsize=fontsize)

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

    def find_entity(
            self, axes: Axes, etype, index=np.s_[:],
            showindex=False,
            color='r', markersize=20,
            fontcolor='k', fontsize=24
        ):
        """
        @brief ?

        @return: The entities found.
        """
        self._get_ploter(axes)

        if not isinstance(self, Mesh):
            raise TypeError(f"Only works on mesh types.")

        bc: NDArray = self.entity_barycenter(etype, index=index).numpy()
        print(etype, bc)

        if bc.ndim == 1:
            bc = bc[:, None]
        if bc.shape[1] == 1:
            bc = np.concatenate([bc, np.zeros_like(bc)], axis=-1)

        if index == np.s_[:]:
            index = np.arange(bc.shape[0])
        elif isinstance(index, np.int_):
            index = np.array([index], dtype=np.int_)
        elif isinstance(index, np.ndarray) and (index.dtype is np.bool_):
            index, = np.nonzero(index)
        elif isinstance(index, list) and isinstance(index[0], np.bool_):
            index, = np.nonzero(index)
        else:
            raise TypeError(f"Unknown index format.")

        if isinstance(color, np.ndarray) and (np.isreal(color[0])):
            mapper = array_color_map(color, 'rainbow')
            color = mapper.to_rgba(color)

        bc = bc[index]

        self._ploter.point_scatter(bc, color=color, markersize=markersize)
        if showindex:
            self._ploter.show_index(bc, number=index, fontcolor=fontcolor,
                                    fontsize=fontsize)
        return bc


_MT = TypeVar('_MT', bound=Mesh)

class MeshCanvas(Generic[_MT]):
    _axes: Axes
    def __init__(self, mesh: _MT) -> None:
        self._mesh = mesh

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

    def show_index(self, points: NDArray, number: NDArray, fontcolor='k', fontsize=24):
        """
        @brief Display index text in the axes.

        @param points: Locations of number texts.
        @param number: Array of numbers to display.
        """
        GD = points.shape[-1]
        if GD == 2:
            for i, idx in enumerate(number):
                self._axes.text(points[i, 0], points[i, 1], str(idx),
                        multialignment='center', fontsize=fontsize,
                        color=fontcolor)

        elif GD == 3:
            for i, idx in enumerate(number):
                self._axes.text(
                        points[i, 0], points[i, 1], points[i, 2],
                        str(idx),
                        multialignment='center',
                        fontsize=fontsize, color=fontcolor)

        else:
            raise ValueError(f"Can only tackle the 'points' array with length\
                             2 or 3 in the last dimension.")

    def show_multi_index(self, points: NDArray, multi_index: NDArray,
                         fontcolor='k', fontsize=24):
        if isinstance(multi_index, np.ndarray) and (multi_index.ndim > 1):
            GD = points.shape[-1]
            if GD == 2:
                for i, idx in enumerate(multi_index):
                    s = str(idx).replace('[', '(')
                    s = s.replace(']', ')')
                    s = s.replace(' ', ',')
                    self._axes.text(points[i, 0], points[i, 1], s,
                            multialignment='center',
                            fontsize=fontsize,
                            color=fontcolor)
            elif GD == 3:
                for i, idx in enumerate(multi_index):
                    s = str(idx).replace('[', '(')
                    s = s.replace(']', ')')
                    s = s.replace(' ', ',')
                    self._axes.text(points[i, 0], points[i, 1], points[i, 2], s,
                            multialignment='center',
                            fontsize=fontsize,
                            color=fontcolor)
            else:
                raise ValueError(f"Can only tackle the 'points' array with length\
                             2 or 3 in the last dimension.")
        else:
            return self.show_index(points=points, number=multi_index,
                                   fontcolor=fontcolor, fontsize=fontsize)

    def point_scatter(self, points: Optional[NDArray], color, markersize):
        """
        @brief Show the node entities in the axes.

        @param points: An NDArray containing the node positions.
        @param color: Color of the node markers.
        @param markersize: The size of the node markers.

        @return: PathCollection.
        """
        if points is None:
            node: NDArray = self._mesh.entity(0).numpy()
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
            raise ValueError(f"Can only tackle the 'points' array with length\
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
            node: NDArray = self._mesh.entity(0).numpy()
        else:
            node = points

        if struct is None:
            cell: NDArray = self._mesh.entity(1).numpy()
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
            node: NDArray = self._mesh.entity(0).numpy()
        else:
            node = points

        if struct is None:
            cell: NDArray = self._mesh.entity(2).numpy()
            if hasattr(self._mesh.ds, 'ccw'):
                cell = cell[..., self._mesh.ds.ccw]
        else:
            cell = struct

        GD = node.shape[-1]

        if GD == 2:
            from matplotlib.collections import PolyCollection
            poly = PolyCollection(node[cell, :])
        elif GD == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(node[cell, :])
        else:
            raise ValueError(f"Can only tackle the 'points' array with length\
                             2 or 3 in the last dimension.")

        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(linewidths)
        poly.set_facecolor(cellcolor)
        return self._axes.add_collection(poly)


class MeshCanvas1d(MeshCanvas[Mesh1d]):
    def __call__(
            self, plot: Axes,
            nodecolor='k', cellcolor='k',
            markersize=20, linewidths=1,
            aspect='equal',
            shownode=True, showaxis=False,
            box=None, **kwargs
        ):
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
            linewidths: float=1.0,
            aspect=None,
            showaxis: bool=False, colorbar: bool=False, colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None, **kwargs
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
        if hasattr(self._mesh.ds, 'ccw'):
            cell = cell[..., self._mesh.ds.ccw]

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
            nodecolor='k', edgecolor='k', cellcolor='w',
            markersize=20, linewidths=0.5, alpha=0.8,
            aspect=[1, 1, 1],
            showaxis=False, shownode=False, showedge=False, threshold=None,
            box=None, **kwargs
        ):
        self._axes = self.get_axes(plot, projection='3d')
        self.set_aspect(aspect)
        self.set_show_axis(showaxis)
        self.set_lim(box)

        if isinstance(nodecolor, np.ndarray) and np.isreal(nodecolor[0]):
            mapper = array_color_map(nodecolor, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node: NDArray = self._mesh.entity('node').numpy()
        if shownode:
            self.point_scatter(points=node, color=nodecolor, markersize=markersize)

        if showedge:
            edge: NDArray = self._mesh.entity('edge').numpy()
            self.length_line(points=node, struct=edge, color=edgecolor,
                             linewidths=linewidths)

        face: NDArray = self._mesh.entity('face').numpy()
        if hasattr(self._mesh.ds, 'ccw'):
            face = face[..., self._mesh.ds.ccw]
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

        poly = self.area_poly(points=node, struct=face, edgecolor=edgecolor,
                       cellcolor=cellcolor, linewidths=linewidths)

        poly.set_alpha(alpha)
        return poly
