from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from matplotlib.collections import PolyCollection, PatchCollection
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.patches import Polygon
from matplotlib.axes import Axes


def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


class MeshPlot():
    def __init__(self, mesh) -> None:
        self._mesh = mesh

    def __call__(
            self, axes: Axes,
            edgecolor='k', cellcolor=[0.5, 0.9, 0.45],
            aspect=None, linewidths: float=1.0,
            showaxis: bool=False, colorbar: bool=False,
            colorbarshrink=1.0,
            cmax=None, cmin=None, cmap='jet', box=None
        ):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.
        """
        self._axes = axes
        self.set_aspect(aspect=aspect)
        self.set_axis(switch=showaxis)
        self.set_lim(box=box)

        if isinstance(cellcolor, Tensor):
            cellcolor = cellcolor.numpy()
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=cmap, cmax=cmax, cmin=cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if colorbar:
                f = self._axes.get_figure()
                f.colorbar(mapper, shrink=colorbarshrink, ax=self._axes)

        poly = self.poly()
        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(linewidths)
        poly.set_facecolor(cellcolor)

        return self._axes.add_collection(poly)

    @staticmethod
    def _arr(array_or_tensor: Union[NDArray, Tensor]) -> NDArray:
        if isinstance(array_or_tensor, Tensor):
            return array_or_tensor.numpy()
        return array_or_tensor

    def set_aspect(self, aspect):
        GD = self._mesh.geo_dimension()

        if (aspect is None) and (GD == 3):
            self._axes.set_box_aspect((1, 1, 1))
            self._axes.set_proj_type('ortho')

        if (aspect is None) and (GD == 2):
            self._axes.set_box_aspect(1)

    def set_axis(self, switch: bool):
        if switch:
            self._axes.set_axis_on()
        else:
            self._axes.set_axis_off()

    def set_lim(self, box: Optional[NDArray]=None):
        """
        @brief Set boundaries for plot area. By default, canvas can contain\
            the whole mesh.
        """
        GD = self._mesh.geo_dimension()
        if box is None:
            node = self._arr(self._mesh.node)
            em = self._mesh.entity_measure('edge')
            tol = np.max(self._arr(em))/100
            box = np.zeros(2*GD, dtype=np.float64)
            box[0::2] = np.min(node, axis=0) - tol
            box[1::2] = np.max(node, axis=0) + tol

        self._axes.set_xlim(box[0:2])
        self._axes.set_ylim(box[2:4])

        if GD == 3:
            self._axes.set_zlim(box[4:6])

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
            poly = PolyCollection(node[cell[:, self._mesh.ds.ccw], :])
        elif GD == 3:
            import mpl_toolkits.mplot3d as a3
            poly = a3.art3d.Poly3DCollection(node[cell[:, self._mesh.ds.ccw], :])
        return poly
