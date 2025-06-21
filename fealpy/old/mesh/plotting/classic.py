
from typing import (
    Callable, Optional, TypeVar, Any, Union, Generic,
    Dict, Type
)

from types import ModuleType
import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes

from ..mesh_base import Mesh
from . import artist as A

_MT = TypeVar('_MT', bound=Mesh)

def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    from matplotlib import colors, cm

    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)

_ploter_map_: Dict[str, Type] = {}

def get_ploter(key: str) -> Type['MeshPloter']:
    if key in _ploter_map_:
        return _ploter_map_[key]
    else:
        raise KeyError(f"Can not find a ploter class that key '{key}' mapping to. "
                       "To use Plotable, register the target ploter first and then "
                       "specify the ploter for mesh by setting a same key. "
                       "See MeshPloter.register() and Plotable.set_ploter().")


class MeshPloter(Generic[_MT]):
    _axes: Axes
    _mesh: _MT

    class PlotArgs():
        def __init__(self, **data: Any) -> None:
            self._data = data

        def __getattr__(self, key):
            return self._data.get(key, None)

        def update_(self, **data: Any):
            self._data.update(data)

    def __init__(self, mesh: _MT) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("MeshPloter only works for mesh type, "
                            f"but got {self.__class__.__name__}.")
        self._mesh = mesh

        # This is the default parameter for ALL Ploters
        self.args = MeshPloter.PlotArgs(
            color='r',                  # color of markers
            nodecolor='k',              # color of nodes
            edgecolor='k',              # color of edges
            facecolor='k',              # color of faces
            cellcolor='k',              # color of cells
            alpha=1.0,                  # alpha channel of cells in 3-d mesh

            marker='o',                 # style of markers
            markersize=20,              # size of markers
            linewidths=1.0,             # width of lines(edges)
            aspect='equal',             # aspect
            showaxis=False,             # show the axis if True
            colorbar=False,             # show color bar if True
            colorbarshrink=1.0,         # colorbarshrink
            cmap='jet',                 # color map

            shownode=True,              # show nodes in 1-d, 3-d mesh if True
            showedge=False,             # show edges in 3-d mesh if True

            etype='cell',               # specify the entity for Finders
            showindex=False,            # show the index of entity for Finders
            index=np.s_[:],             # specify the index of entity to plot

            fontcolor='k',              # color of text
            fontsize=24                 # size of text
        )

    @property
    def current_axes(self):
        return self._axes

    @current_axes.setter
    def current_axes(self, axes: Axes):
        if isinstance(axes, Axes):
            self._axes = axes
        else:
            raise TypeError("Param 'axes' should be Axes type, "
                            f"but got {axes.__class__.__name__}.")

    @property
    def mesh(self):
        return self._mesh

    def _call_impl(self, axes: Union[Axes, ModuleType], *args, **kwargs):
        if isinstance(axes, ModuleType):
            fig = axes.figure()
            axes = fig.add_subplot(1, 1, 1)

        self.current_axes = axes
        self.args.update_(**kwargs)

        return self.draw()

    __call__ = _call_impl

    draw: Callable[..., Any]

    @classmethod
    def register(cls, key: str):
        """
        @brief Register this ploter with a unique string key.
        """
        if key in _ploter_map_:
            ploter = _ploter_map_[key]
            raise KeyError(f"Key '{key}' has been used by ploter {ploter.__name__}.")
        elif not issubclass(cls, MeshPloter):
            raise TypeError(f"Expect subclass of MeshPloter but got itself.")
        elif not isinstance(key, str):
            raise TypeError("Only accepts a single string as the key, "
                            f"but got {key.__class__.__name__}.")
        _ploter_map_[key] = cls

    def set_show_axis(self, switch: bool=True):
        if switch:
            self._axes.set_axis_on()
        else:
            self._axes.set_axis_off()

    def set_lim(self, box: Optional[NDArray]=None, tol=0.1):
        from mpl_toolkits.mplot3d import Axes3D

        GD = self._mesh.geo_dimension()

        if box is None:
            node: NDArray = self._mesh.entity('node')
            if node.ndim == 1:
                node = node.reshape(-1, 1)

            box = np.array([-0.5, 0.5]*3, dtype=np.float64)
            box[0:2*GD:2] = np.min(node, axis=0) - tol
            box[1:1+2*GD:2] = np.max(node, axis=0) + tol

        self._axes.set_xlim(box[0:2])
        self._axes.set_ylim(box[2:4])

        if isinstance(self._axes, Axes3D):
            self._axes.set_zlim(box[4:6])


class EntityFinder(MeshPloter):
    def draw(self):
        """
        @brief Show the barycenter of each entity.
        """
        from ..plotting import artist as A
        from ..plotting.classic import array_color_map

        axes = self.current_axes
        args = self.args
        etype_or_node = args.etype

        if isinstance(etype_or_node, (int, str)):
            # NOTE: Here we slice the entity after generated, because the fucking
            # uniform mesh does not implement the index arg in entity_barycenter.
            # The index arg can be accept, but ignored.
            bc = self.mesh.entity_barycenter(etype=etype_or_node)[args.index]
        elif isinstance(etype_or_node, np.ndarray):
            bc = etype_or_node
        else:
            raise TypeError(f"Invalid entity type or node info.")
        if bc.ndim == 1:
            bc = bc[:, None]

        color = args.color
        if isinstance(color, np.ndarray) and np.isreal(color[0]):
            mapper = array_color_map(color, 'rainbow')
            color = mapper.to_rgba(color)

        A.scatter(axes=axes, points=bc, color=color,
                  marker=args.marker, markersize=args.markersize)
        if args.showindex:
            if args.multiindex is None:
                A.show_index(axes=axes, location=bc, number=args.index,
                            fontcolor=args.fontcolor, fontsize=args.fontsize)
            else:
                A.show_multi_index(axes=axes, location=bc, text_list=args.multiindex,
                                   fontcolor=args.fontcolor, fontsize=args.fontsize)

EntityFinder.register('finder')


##################################################
### MeshPloter subclasses
##################################################


class AddPlot1d(MeshPloter):
    def draw(self):
        axes = self.current_axes
        args = self.args
        self.set_lim(args.box)
        axes.set_aspect(args.aspect)
        self.set_show_axis(args.showaxis)

        node: NDArray = self.mesh.entity('node')

        if node.ndim == 1:
            node = node[:, None]

        if args.shownode:
            A.scatter(axes=axes, points=node, color=args.nodecolor,
                      markersize=args.markersize)

        cell = self.mesh.entity('cell')

        return A.line(axes=axes, points=node, struct=cell, color=args.cellcolor,
                      linewidths=args.linewidths)

AddPlot1d.register('1d')


class AddPlot2dHomo(MeshPloter):
    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        self.args.update_(cellcolor='#99BBF6')

    def draw(self):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.
        """
        axes = self.current_axes
        args = self.args
        self.set_lim(args.box)
        axes.set_aspect(args.aspect)
        self.set_show_axis(args.showaxis)

        cellcolor = args.cellcolor
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=args.cmap,
                                     cmax=args.cmax, cmin=args.cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if args.colorbar:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=args.colorbarshrink, ax=self._axes)

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        ccw = getattr(self.mesh.ds, 'ccw', None)
        if ccw is not None:
            cell = cell[..., ccw]

        return A.poly(axes=axes, points=node, struct=cell,
                      edgecolor=args.edgecolor, cellcolor=cellcolor,
                      linewidths=args.linewidths, alpha=args.alpha)

AddPlot2dHomo.register('2d')


class AddPlot2dPoly(MeshPloter):
    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        self.args.update_(cellcolor='#99BBF6')

    def draw(self):
        """
        @brief Add a mesh canvas to the given axes. Then the entities of mesh\
               (i.e. node, cell) can be found and shown in the mesh background.
        """
        axes = self.current_axes
        args = self.args
        self.set_lim(args.box)
        axes.set_aspect(args.aspect)
        self.set_show_axis(args.showaxis)

        cellcolor = args.cellcolor
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=args.cmap,
                                     cmax=args.cmax, cmin=args.cmin)
            cellcolor = mapper.to_rgba(cellcolor)
            if args.colorbar:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=args.colorbarshrink, ax=self._axes)

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell') # A list containing indices of vertices.

        return A.poly_(axes=axes, points=node, struct_seq=cell,
                       edgecolor=args.edgecolor, cellcolor=cellcolor,
                       linewidths=args.linewidths, alpha=args.alpha)

AddPlot2dPoly.register('polygon2d')


class AddPlot3dHomo(MeshPloter):
    def __init__(self, mesh: Any) -> None:
        super().__init__(mesh)
        self.args.update_(cellcolor='w', alpha=0.8, linewidth=0.5)

    def draw(self):
        axes = self.current_axes
        args = self.args
        self.set_lim(args.box)
        axes.set_aspect(args.aspect)
        self.set_show_axis(args.showaxis)

        nodecolor = args.nodecolor
        if isinstance(nodecolor, np.ndarray) and np.isreal(nodecolor[0]):
            mapper = array_color_map(nodecolor, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node = self.mesh.entity('node')
        if args.shownode:
            A.scatter(axes=axes, points=node, color=nodecolor, markersize=args.markersize)

        if args.showedge:
            edge = self.mesh.entity('edge')
            A.line(axes=axes, points=node, struct=edge, color=args.edgecolor,
                   linewidths=args.linewidths)

        face = self.mesh.entity('face')
        ccw = getattr(self.mesh.ds, 'ccw', None)
        if ccw is not None:
            face = face[..., ccw]
        isBdFace = self.mesh.ds.boundary_face_flag()

        if args.threshold is None:
            face = face[isBdFace]
        else:
            bc = self.mesh.entity_barycenter('cell')
            isKeepCell = args.threshold(bc)
            face2cell = self.mesh.ds.face_to_cell()
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) and isBdFace
            face = face[isBdFace | isInterfaceFace]

        return A.poly(axes=axes, points=node, struct=face, edgecolor=args.edgecolor,
                      cellcolor=args.cellcolor, linewidths=args.linewidths,
                      alpha=args.alpha)

AddPlot3dHomo.register('3d')
