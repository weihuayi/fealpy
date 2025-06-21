"""
Provide plotting tools for given points with structure.
"""

from typing import Sequence, Union, Any
from numpy.typing import NDArray
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def _validate_geo_dim(axes: Axes, points: NDArray):
    if points.ndim != 2:
        raise ValueError("Arg 'points' should have 2 dimensions.")

    (NN, GD) = points.shape

    if isinstance(axes, Axes3D):
        if GD == 3:
            return points
        elif GD in {1, 2}:
            tailer = np.zeros((NN, 3-GD), dtype=points.dtype)
            return np.concatenate([points, tailer], axis=-1)

    elif isinstance(axes, Axes):
        if GD == 3:
            raise ValueError("Can not plot 3d points in 2d axes.")
        elif GD == 2:
            return points
        elif GD == 1:
            tailer = np.zeros((NN, 2-GD), dtype=points.dtype)
            return np.concatenate([points, tailer], axis=-1)

    raise ValueError(f"Plotting for {GD}-d points has not been implemented.")


def show_index(axes: Axes, location: NDArray, number: Union[NDArray, slice],
               fontcolor='k', fontsize=24):
    """
    @brief Show index numbers in the given locations in 2d or 3d.
    """
    if isinstance(number, np.ndarray):
        if number.dtype == np.bool_:
            number, = np.nonzero(number)
    elif isinstance(number, slice):
        number = np.arange(location.shape[0])[number]
    else:
        raise TypeError("Unknown index number format.")

    GD = location.shape[-1]
    loc = _validate_geo_dim(axes, location)
    if GD == 3:
        for i, idx in enumerate(number):
            axes.text(loc[i, 0], loc[i, 1], loc[i, 2], str(idx),
                multialignment='center', fontsize=fontsize,
                color=fontcolor)

    else:
        for i, idx in enumerate(number):
            axes.text(
                loc[i, 0], loc[i, 1], str(idx),
                multialignment='center',
                fontsize=fontsize, color=fontcolor)


def show_multi_index(axes: Axes, location: NDArray, text_list: Sequence[Any],
                     fontcolor='k', fontsize=14):
    GD = location.shape[-1]
    loc = _validate_geo_dim(axes, location)
    if GD == 3:
        for i, text in enumerate(text_list):
            axes.text(loc[i, 0], loc[i, 1], loc[i, 2], text,
                multialignment='center', fontsize=fontsize,
                color=fontcolor)

    else:
        for i, text in enumerate(text_list):
            axes.text(
                loc[i, 0], loc[i, 1], text,
                multialignment='center',
                fontsize=fontsize, color=fontcolor)


def scatter(axes: Axes, points: NDArray, color,
            marker: str='o', markersize: float=12):
    """
    @brief Show points in the axes.

    @param points: An NDArray[float] containing positions with shape (NN, GD) where\
                   NN is the number of points and GD is their geometry dimension.
    @param color: Color of the node markers.
    @param markersize: The size of the node markers.

    @return: PathCollection.
    """
    GD = points.shape[-1]
    loc = _validate_geo_dim(axes, points)

    if GD == 3:
        return axes.scatter(
            loc[:, 0], loc[:, 1], loc[:, 2],
            color=color, s=markersize, marker=marker
        ),
    else:
        return axes.scatter(
            loc[:, 0], loc[:, 1],
            color=color, s=markersize, marker=marker
        )


def line(axes: Axes, points: NDArray, struct: NDArray, color, linewidths):
    """
    @brief Show lines in the axes.

    @param points: An NDArray[float] containing positions of the vertices.
    @param struct: An NDArray[int] of indices of vertices in two ends of an edge.\
                   It should be with shape (NE, 2) where NE matches the number of\
                   edges and 2 means there are two ends in each edge.
    @param color: Color of lines.
    @param linewidths: Widths of lines.

    @return: LineCollection or Line3DCollection.
    """
    GD = points.shape[-1]
    loc = _validate_geo_dim(axes, points)
    vts = loc[struct, :]

    if GD == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lines = Line3DCollection(vts, linewidths=linewidths, colors=color)
    else:
        from matplotlib.collections import LineCollection
        lines = LineCollection(vts, linewidths=linewidths, colors=color)

    return axes.add_collection(lines)


def poly(axes: Axes, points: NDArray, struct: NDArray, edgecolor,
         cellcolor, linewidths=0.1, alpha=1.0):
    """
    @brief Show homogeneous polygons in the axes.

    @param points: An NDArray[float] containing positions of the vertices.
    @param struct: An NDArray[int] of indices of nodes in every polygons.\
                   It should be with shape (NC, NVC) where NC matches the number\
                   cells and NVC(>=3) is the number of vertices in each cell.

    @return: PolyCollection or Poly3DCollection.
    """
    GD = points.shape[-1]
    loc = _validate_geo_dim(axes, points)
    vts = loc[struct, :]

    if GD == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection(vts)
    else:
        from matplotlib.collections import PolyCollection
        poly = PolyCollection(vts)

    poly.set_edgecolor(edgecolor)
    poly.set_linewidth(linewidths)
    poly.set_facecolor(cellcolor)
    poly.set_alpha(alpha)

    return axes.add_collection(collection=poly)


def poly_(axes: Axes, points: NDArray, struct_seq: Sequence[NDArray],
          edgecolor, cellcolor, linewidths=0.1, alpha=1.0):
    """
    @brief Show polygons (may have different shape of cells) in the axes.
    """
    GD = points.shape[-1]
    if GD != 2:
        raise NotImplementedError('Polygons with points with geometry dimension'
                                  f'{GD} has not been implemented.')
    if not isinstance(axes, Axes):
        raise TypeError(f"Require 2d Axes object but got {axes.__class__.__name__}.")

    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    NC = len(struct_seq)
    patches = [
        Polygon(points[struct_seq[i], :], closed=True)
        for i in range(NC)
    ]
    poly = PatchCollection(patches=patches)

    poly.set_edgecolor(edgecolor)
    poly.set_linewidth(linewidths)
    poly.set_facecolor(cellcolor)
    poly.set_alpha(alpha)

    return axes.add_collection(collection=poly)
