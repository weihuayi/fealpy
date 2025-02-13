"""
Provide the `add_plot` API in Plotable.
"""

from typing import Optional, Union, Iterable, Any

from ..backend import TensorLike

S_ = slice(None)
Index = Union[slice, TensorLike]


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
            node: Optional[TensorLike]=None,
            index: Index=S_,
            showindex=False,
            multiindex: Optional[Iterable[Any]]=None,
            color='#CC0000', marker='o', markersize=12,
            fontsize=12, fontcolor='#CC0000'):
        """Show barycenters of nodes in the axes.

        Parameters:
            axes (Axes): The axes to draw points.\n
            node (NDArray, optional): an array containing node to draw. If not provided,\
                use the nodes in the mesh.\n
            index (NDArray | slice): an array or slice of the nodes.\n
            showindex (bool, optional): Print indices of entities if `True`. Defaults to False.\n
            multiindex (Iterable | None, optional): A iterable serves as the index\
                text to be printed when `showindex == True`.\n
            color (str | NDArray, optional): Color of node points scattered.\
                A string such as `'r'`, `'b'` or a float sequence containing RGB information,\
                like `[0.2, 0.4, 0.6]` are acceptable.\n
            markersize (float, optional): The size of node points scattered, defualts to 20.0.\n
            fontsize (int, optional): The size of indices printed, defaults to 16.\n
            fontcolor (str | Sequence[float], optional): Color of indices font.

        Returns:
            Collection: matplotlib.collections.Collection
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
                    showindex=showindex, multiindex=multiindex,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    fontsize=fontsize,
                    fontcolor=fontcolor)

    def find_edge(self, axes, *,
            index=S_,
            showindex=False, multiindex=None,
            color='#00CC3A', marker='^', markersize=15,
            fontsize=14, fontcolor='#00CC3A'):
        """Show barycenters of edges in the axes."""
        return self.find_entity(
                axes, etype='edge', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_face(self, axes, *,
            index=S_,
            showindex=False, multiindex=None,
            color='#CCA000', marker='d', markersize=18,
            fontsize=16, fontcolor='#CCA000'):
        """Show barycenters of faces in the axes."""
        return self.find_entity(
                axes, etype='face', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)

    def find_cell(self, axes, *,
            index=S_,
            showindex=False, multiindex=None,
            color='#0012CC', marker='s', markersize=21,
            fontsize=18, fontcolor='#0012CC'):
        """Show barycenters of cells in the axes."""
        return self.find_entity(
                axes, etype='cell', index=index,
                showindex=showindex, multiindex=multiindex,
                color=color,
                marker=marker,
                markersize=markersize,
                fontsize=fontsize,
                fontcolor=fontcolor)
    @property
    def show_angle(self):
        from ..plotting.classic import get_ploter
        return get_ploter('show_angle')(self)
