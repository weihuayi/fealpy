
from typing import (
    Callable, Iterable, Sequence, Dict,
    Any, Optional, Union, Type,
    overload, Generic, TypeVar,
)
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.collections import Collection

from .. import logger
from ..mesh import Mesh, HomogeneousMesh
from ..backend import backend_manager as bm
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
    mesh: _MT

    def __init__(self, mesh: _MT) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("MeshPloter only works for Mesh type, "
                            f"but got {self.__class__.__name__}.")
        self.mesh = mesh

        # This is the default parameter for ALL Ploters
        self._args = dict(
            color = 'r',                  # color of markers
            nodecolor = 'k',              # color of nodes
            edgecolor = 'k',              # color of edges
            facecolor = 'k',              # color of faces
            cellcolor = 'k',              # color of cells
            alpha = 1.0,                  # alpha channel of cells in 3-d mesh
            marker = 'o',                 # style of markers
            markersize = 20,              # size of markers
            linewidths = 0.75,            # width of lines(edges)
            aspect = 'equal',             # aspect
            box = None,                   # box of axes

            showaxis = False,             # show the axis if True
            colorbar = False,             # show color bar if True
            colorbarshrink = 1.0,         # colorbarshrink
            cmap = 'jet',                 # color map
            cmax = None,
            cmin = None,
            shownode = True,              # show nodes in 1-d, 3-d mesh if True
            showedge = False,             # show edges in 3-d mesh if True

            etype = 'cell',               # specify the entity for Finders
            showindex = False,            # show the index of entity for Finders
            multiindex = None,
            index = slice(None),          # specify the index of entity to plot
            threshold = None,

            fontcolor = 'k',              # color of text
            fontsize = 24                 # size of text
        )

    def _call_impl(self, axes: Union[Axes, ModuleType], *args, **kwargs):
        if isinstance(axes, ModuleType):
            fig = axes.figure()
            axes = fig.add_subplot(1, 1, 1)

        self._args.update(**kwargs)

        return self.draw(axes, *args, **self._args)

    @overload # type hints
    def __call__(self, axes: Union[Axes, ModuleType], *,
                color: str = ...,                  # color of markers
                nodecolor: str = ...,              # color of nodes
                edgecolor: str = ...,              # color of edges
                facecolor: str = ...,              # color of faces
                cellcolor: str = ...,              # color of cells
                alpha: float = ...,                # alpha channel of cells in 3-d mesh
                marker: str = ...,                 # style of markers
                markersize: float = ...,           # size of markers
                linewidths: float = ...,           # width of lines(edges)
                aspect = 'equal',             # aspect
                box: Sequence[float] = ...,

                showaxis = False,             # show the axis if True
                colorbar = False,             # show color bar if True
                colorbarshrink = 1.0,         # colorbarshrink
                cmap = 'jet',                 # color map

                shownode = True,              # show nodes in 1-d, 3-d mesh if True
                showedge = False,             # show edges in 3-d mesh if True

                etype: Any = ...,             # specify the entity for Finders
                showindex = False,            # show the index of entity for Finders
                multiindex: Iterable[Any] = ...,
                index: slice = ...,           # specify the index of entity to plot
                threshold: Callable = ...,

                fontcolor = 'k',              # color of text
                fontsize = 24) -> Collection: ...
    __call__ = _call_impl

    draw: Callable[..., Any]

    @classmethod
    def register(cls, key: str, /) -> None:
        """Register this ploter with a unique string key."""
        if key in _ploter_map_:
            ploter = _ploter_map_[key]
            raise KeyError(f"Key '{key}' has been used by ploter {ploter.__name__}.")
        elif not issubclass(cls, MeshPloter):
            raise TypeError(f"Expect subclass of MeshPloter but got itself.")
        elif not isinstance(key, str):
            raise TypeError("Only accepts a single string as the key, "
                            f"but got {key.__class__.__name__}.")
        _ploter_map_[key] = cls

    @staticmethod
    def set_show_axis(axes: Axes, switch: bool=True):
        if switch:
            axes.set_axis_on()
        else:
            axes.set_axis_off()

    def set_lim(self, axes: Axes, box: Optional[NDArray]=None, tol=0.1):
        from mpl_toolkits.mplot3d import Axes3D

        GD = self.mesh.geo_dimension()

        if box is None:
            node: NDArray = bm.to_numpy(self.mesh.entity('node'))
            if node.ndim == 1:
                node = node.reshape(-1, 1)

            box = np.array([-0.5, 0.5]*3, dtype=np.float64)
            box[0:2*GD:2] = np.min(node, axis=0) - tol
            box[1:1+2*GD:2] = np.max(node, axis=0) + tol

        axes.set_xlim(box[0:2])
        axes.set_ylim(box[2:4])

        if isinstance(axes, Axes3D):
            axes.set_zlim(box[4:6])


class EntityFinder(MeshPloter[_MT]):
    def draw(self, axes: Axes, *args, **kwargs):
        """Show the barycenter of each entity."""
        from ..plotting import artist as A

        etype_or_node = kwargs['etype']
        color = kwargs['color']

        if isinstance(etype_or_node, (int, str)):
            bc = self.mesh.entity_barycenter(etype=etype_or_node, index=kwargs['index'])
        elif isinstance(etype_or_node, np.ndarray):
            bc = etype_or_node
        else:
            raise TypeError(f"Invalid entity type or node info.")

        bc: NDArray = bm.to_numpy(bc)

        if bc.ndim == 1:
            bc = bc[:, None]

        if isinstance(color, np.ndarray) and np.isreal(color[0]):
            mapper = array_color_map(color, 'rainbow')
            color = mapper.to_rgba(color)

        coll = A.scatter(axes=axes, points=bc, color=color,
                         marker=kwargs['marker'], markersize=kwargs['markersize'])

        if kwargs['showindex']:
            if kwargs['multiindex'] is None:
                A.show_index(axes=axes, location=bc, number=kwargs['index'],
                            fontcolor=kwargs['fontcolor'], fontsize=kwargs['fontsize'])
            else:
                A.show_multi_index(axes=axes, location=bc, text_list=kwargs['multiindex'],
                                   fontcolor=kwargs['fontcolor'], fontsize=kwargs['fontsize'])

        return coll

EntityFinder.register('finder')


##################################################
### MeshPloter subclasses
##################################################


class AddPlot1d(MeshPloter[HomogeneousMesh]):
    def draw(self, axes: Axes, *args, **kwargs):
        self.set_lim(axes, kwargs['box'])
        axes.set_aspect(kwargs['aspect'])
        self.set_show_axis(axes, kwargs['showaxis'])

        node = bm.to_numpy(self.mesh.entity('node'))

        if node.ndim == 1:
            node = node[:, None]

        if kwargs['shownode']:
            A.scatter(axes=axes, points=node, color=kwargs['nodecolor'],
                      markersize=kwargs['markersize'])

        cell = bm.to_numpy(self.mesh.entity('cell'))

        return A.line(axes=axes, points=node, struct=cell, color=kwargs['cellcolor'],
                      linewidths=kwargs['linewidths'])

AddPlot1d.register('1d')


class AddPlot2dHomo(MeshPloter[HomogeneousMesh]):
    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._args.update(cellcolor='#E3F1FF')

    def draw(self, axes: Axes, *args, **kwargs):
        self.set_lim(axes, kwargs['box'])
        axes.set_aspect(kwargs['aspect'])
        self.set_show_axis(axes, kwargs['showaxis'])

        cellcolor = kwargs['cellcolor']
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=kwargs['cmap'],
                                     cmax=kwargs['cmax'], cmin=kwargs['cmin'])
            cellcolor = mapper.to_rgba(cellcolor)
            if kwargs['colorbar']:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=kwargs['colorbarshrink'], ax=axes)

        node = bm.to_numpy(self.mesh.entity('node'))
        cell = bm.to_numpy(self.mesh.entity('cell'))

        if hasattr(self.mesh, 'ccw'):
            ccw = bm.to_numpy(self.mesh.ccw)
            cell = cell[:, ccw]
        else:
            logger.warning("The mesh has no attribute `ccw`, the local vertices "
                           "order of the 2-d entity (cell) is not determined.")

        return A.poly(axes=axes, points=node, struct=cell,
                      edgecolor=kwargs['edgecolor'], cellcolor=cellcolor,
                      linewidths=kwargs['linewidths'], alpha=kwargs['alpha'])

AddPlot2dHomo.register('2d')


class AddPlot2dPoly(MeshPloter[HomogeneousMesh]):
    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._args.update(cellcolor='#E3F1FF')

    def draw(self, axes: Axes, *args, **kwargs):
        self.set_lim(axes, kwargs['box'])
        axes.set_aspect(kwargs['aspect'])
        self.set_show_axis(axes, kwargs['showaxis'])

        cellcolor = kwargs['cellcolor']
        if isinstance(cellcolor, np.ndarray) and np.isreal(cellcolor[0]):
            mapper = array_color_map(cellcolor, cmap=kwargs['cmap'],
                                     cmax=kwargs['cmax'], cmin=kwargs['cmin'])
            cellcolor = mapper.to_rgba(cellcolor)
            if kwargs['colorbar']:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=kwargs['colorbarshrink'], ax=axes)

        node = bm.to_numpy(self.mesh.entity('node'))
        cell, loc = self.mesh.entity('cell')
        cell_list = [cell[loc[i]:loc[i+1]] for i in range(loc.shape[0]-1)]

        return A.poly_(axes=axes, points=node, struct_seq=cell_list,
                       edgecolor=kwargs['edgecolor'], cellcolor=cellcolor,
                       linewidths=kwargs['linewidths'], alpha=kwargs['alpha'])

AddPlot2dPoly.register('polygon2d')


class AddPlot3dHomo(MeshPloter[HomogeneousMesh]):
    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._args.update(cellcolor='w', alpha=0.5)

    def draw(self, axes: Axes, *args, **kwargs):
        self.set_lim(axes, kwargs['box'])
        axes.set_aspect(kwargs['aspect'])
        self.set_show_axis(axes, kwargs['showaxis'])

        nodecolor = kwargs['nodecolor']
        if isinstance(nodecolor, np.ndarray) and np.isreal(nodecolor[0]):
            mapper = array_color_map(nodecolor, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node = bm.to_numpy(self.mesh.entity('node'))
        face = bm.to_numpy(self.mesh.entity('face'))
        isBdFace = bm.to_numpy(self.mesh.boundary_face_flag())

        if kwargs['shownode']:
            A.scatter(axes=axes, points=node, color=nodecolor, markersize=kwargs['markersize'])

        if kwargs['showedge']:
            edge = bm.to_numpy(self.mesh.entity('edge'))
            A.line(axes=axes, points=node, struct=edge, color=kwargs['edgecolor'],
                   linewidths=kwargs['linewidths'])

        if hasattr(self.mesh, 'ccw'):
            ccw = bm.to_numpy(self.mesh.ccw)
            face = face[:, ccw]
        else:
            logger.warning("The mesh has no attribute `ccw`, the local vertices "
                           "order of the 2-d entity (face) is not determined.")

        if kwargs['threshold'] is None:
            face = face[isBdFace]
        else:
            bc = bm.to_numpy(self.mesh.entity_barycenter('cell'))
            isKeepCell = kwargs['threshold'](bc)
            face2cell = self.mesh
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) and isBdFace
            face = face[isBdFace | isInterfaceFace]

        return A.poly(axes=axes, points=node, struct=face, edgecolor=kwargs['edgecolor'],
                      cellcolor=kwargs['cellcolor'], linewidths=kwargs['linewidths'],
                      alpha=kwargs['alpha'])

AddPlot3dHomo.register('3d')
