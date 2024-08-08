"""
Provide the `add_plot` API in Plotable.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class Plotable():
    """Base class for plotable meshes. Inherit this class to obtain several\
    plotting methods.

    Before using these plotting methods, call the class method
    `MeshClass.set_ploter()` to chose a proper ploter for the mesh type.
    For example, specity a plotter for a mesh type named 'TriangleMesh':
    ```
        class TriangleMesh(Mesh2d, Plotable):
            # codes for mesh ...
            ...
        TriangleMesh.set_ploter('unique str key of the ploter')
    ```
    Here `MyPloter` is a subclass of `MeshPloter`. Then, instances of `TriangleMesh`
    may call methods such as `add_plot()` to draw the mesh.

    See Also:
        MeshPloter.
    """
    _ploter_class: Optional[str] = None

    @classmethod
    def set_ploter(cls, ploter: str):
        cls._ploter_class = ploter

    @property
    def add_plot(self):
        """Show the mesh."""
        from ..plotting.classic import get_ploter
        return get_ploter(self._ploter_class)(self)

    @property
    def find_entity(self):
        """Show the barycenter of each entity."""
        from ..plotting.classic import get_ploter
        return get_ploter('finder')(self)

### set different default values for entities

    def find_node(self, axes, *,
            node: Optional[NDArray]=None,
            index=np.s_[:],
            multiindex=None,
            showindex=False,
            color='#CC0000', marker='o', markersize=12,
            fontsize=12, fontcolor='#CC0000'):
        """Show nodes in the axes.

        Parameters:
            axes (Axes): The axes to draw points.
            node (NDArray): an array containing node to draw, optional. If not provided,\
                use the nodes in the mesh.
            index (NDArray | slice): an array or slice controlling the ID of nodes and providing\
                index to be printed when `showindex == True`.
            showindex (bool): Print indices of entities if `True`.
            color (str | NDArray): Color of node points scattered, defualts to `'r'`.\
                A string such as `'r'`, `'b'` or a float sequence containing RGB information,
                like `[0.2, 0.4, 0.6]` are acceptable.
            markersize (float): The size of node points scattered, defualts to 20.0.
            fontsize (int): The size of indices printed, defaults to 16.
            fontcolor (str | Sequence[float]): Color of indices font.
        """
        if node is None:
            return self.find_entity(
                    axes, etype='node', index=index,
                    showindex=showindex, multiindex=multiindex,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)
        else:
            return self.find_entity(
                    axes, etype=node, index=index,
                    showindex=showindex,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)

    def find_edge(self, axes, *,
            index=np.s_[:],
            showindex=False, multiindex=None,
            color='#00CC3A', marker='^', markersize=15,
            fontsize=14, fontcolor='#00CC3A'):
        return self.find_entity(
                axes, etype='edge', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_face(self, axes, *,
            index=np.s_[:],
            showindex=False, multiindex=None,
            color='#CCA000', marker='d', markersize=18,
            fontsize=16, fontcolor='#CCA000'):
        return self.find_entity(
                axes, etype='face', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_cell(self, axes, *,
            index=np.s_[:],
            showindex=False, multiindex=None,
            color='#0012CC', marker='s', markersize=21,
            fontsize=18, fontcolor='#0012CC'):
        return self.find_entity(
                axes, etype='cell', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)
