"""
Provide the `add_plot` API in Plotable.
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from .mesh import Mesh


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

        TriangleMesh.set_ploter('unique str key of the ploter')
    ```
    Here `MyPloter` is a subclass of `MeshPloter`. Then, instances of `TriangleMesh`
    may call methods such as `add_plot()` to draw the mesh.

    @seealso: MeshPloter.
    """
    _ploter_class: Optional[str] = None

    @classmethod
    def set_ploter(cls, ploter: str):
        cls._ploter_class = ploter

    @property
    def add_plot(self):
        """
        @brief Show the mesh.
        """
        from ..plotting.classic import get_ploter
        return get_ploter(self._ploter_class)(self)

    @property
    def find_entity(self):
        """
        @brief Show the barycenter of each entity.
        """
        from ..plotting.classic import get_ploter
        return get_ploter('finder')(self)

    def find_node(self, axes, *,
            node: Optional[NDArray]=None,
            index=np.s_[:],
            showindex=False,
            color='r', markersize=20,
            fontsize=16, fontcolor='r',
            multi_index=None) -> None:
        """
        @brief Show nodes in the axes.

        @param axes: The axes to draw points.
        @param node: an array containing node to draw, optional. If not provided,\
        use the nodes in the mesh.
        @param index: an array or slice controlling the ID of nodes and providing\
        index to be printed when `showindex == True`.
        @param showindex: bool. Print indices of entities if `True`.
        @param color: str | NDArray. Color of node points scattered, defualts to `'r'`.\
        A string such as `'r'`, `'b'` or a float sequence containing RGB information,
        like `[0.2, 0.4, 0.6]` are acceptable.
        @param markersize: float. The size of node points scattered, defualts to 20.0.
        @param fontsize: int. The size of indices printed, defaults to 16.
        @param fontcolor: str | Sequence[float]. Color of indices font.
        """
        if node is None:
            return self.find_entity(
                    axes, etype_or_node='node', index=index,
                    showindex=showindex,
                    color=color,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)
        else:
            return self.find_entity(
                    axes, etype_or_node=node, index=index,
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
