"""
Provide the `add_plot` API in Plotable.
"""

from typing import Optional, Union
import numpy as np
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

    @property
    def add_plot(self):
        if not isinstance(self, Mesh):
            raise TypeError("Plotable only works for mesh type,"
                            f"but got {self.__class__.__name__}.")

        from ..plotting.classic import get_ploter

        if self._ploter_class is not None:
            return get_ploter(self._ploter_class)(self)
        else:
            raise Exception('MeshPloter of the type of mesh should be specified'
                            'before drawing. If a mesh is inherited from Plotable,'
                            'use MeshClass.set_ploter(MeshPloterClass) to specify.')

    @classmethod
    def set_ploter(cls, ploter: str):
        cls._ploter_class = ploter

    def find_entity(self, axes, etype: Union[int, str], index=np.s_[:],
                    showindex: bool=False, color='r', markersize=20,
                    fontcolor='k', fontsize=24):
        """
        @brief Show the barycenter of each entity.
        """
        from ..plotting import artist as A
        from ..plotting.classic import array_color_map

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
