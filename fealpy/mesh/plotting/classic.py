
from collections.abc import Sequence
from typing import Any, Optional, overload
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.collections import Collection

from ..view import Mesh
from . import artist as A

__all__ = [
    'MeshPloter',
    'EntityFinder'
]

_ENTITY_NAMES = {'node', 'edge', 'tri', 'quad', 'tet', 'hex', 'prism', 'pyramid'}
_ENTITY_ALIASES = {
    'cell': 'cell', 'CELL': 'cell', 'Cell': 'cell',
    'face': 'face', 'FACE': 'face', 'Face': 'face',
    'edge': 'edge', 'EDGE': 'edge', 'Edge': 'edge',
    'node': 'node', 'NODE': 'node', 'Node': 'node',
    'vertex': 'node', 'VERTEX': 'node', 'Vertex': 'node',
}


def array_color_map(arr: NDArray, cmap,
                    cmax: Optional[float]=None, cmin: Optional[float]=None):
    from matplotlib import colors, cm

    cmax = cmax or arr.max()
    cmin = cmin or arr.min()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


class MeshPloter:
    mesh: Mesh

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

        self._args = dict(
            nodecolor = 'r',
            edgecolor = 'k',
            cellcolor = 'g',
            alpha = 1.0,
            marker = 'o',
            markersize = 20,
            linewidths = 0.75,
            aspect = 'equal',
            box = None,
            showaxis = False,
            entity = None,
            index = slice(None),
        )

    @overload
    def __call__(self, axes: Axes | ModuleType, *,
                nodecolor: str = ...,
                edgecolor: str = ...,
                cellcolor: str = ...,
                alpha: float = ...,
                marker: str = ...,
                markersize: float = ...,
                linewidths: float = ...,
                aspect = 'equal',
                box: Sequence[float] = ...,
                showaxis = False,
                entity: Any = ...,
                index: slice = ...) -> list[Collection]: ...
    @overload
    def __call__(self, axes: Axes | ModuleType, *args, **kwargs) -> list[Collection]: ...
    def __call__(self, axes: Axes | ModuleType, *args, **kwargs):
        if isinstance(axes, ModuleType):
            fig = axes.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = axes

        self._args.update(**kwargs)
        return self.draw(ax, *args, **self._args)

    @staticmethod
    def set_show_axis(axes: Axes, switch: bool = True) -> None:
        if switch:
            axes.set_axis_on()
        else:
            axes.set_axis_off()

    def _node_array(self) -> NDArray:
        node = np.asarray(self.mesh.block.positions)
        if node.ndim == 1:
            node = node.reshape(-1, 1)
        return node

    def set_lim(self, axes: Axes, box: Optional[NDArray]=None, tol=0.1) -> None:
        from mpl_toolkits.mplot3d import Axes3D

        GD = self.mesh.geo_dimension()

        if box is None:
            node = self._node_array()
            box = np.array([-0.5, 0.5]*3, dtype=np.float64)
            box[0:2*GD:2] = np.min(node, axis=0) - tol
            box[1:1+2*GD:2] = np.max(node, axis=0) + tol

        axes.set_xlim(box[0:2])
        axes.set_ylim(box[2:4])

        if isinstance(axes, Axes3D):
            axes.set_zlim(box[4:6])

    def _draw_nodes(self, axes: Axes, node: NDArray, kwargs) -> list[Collection]:
        return [A.scatter(axes=axes, points=node[kwargs['index']], color=kwargs['nodecolor'],
                          marker=kwargs['marker'], markersize=kwargs['markersize'])]

    def _draw_edges(self, axes: Axes, node: NDArray, indices: NDArray, kwargs) -> list[Collection]:
        if indices.size == 0:
            return []
        return [A.line(axes=axes, points=node, struct=indices[kwargs['index']],
                       color=kwargs['edgecolor'], linewidths=kwargs['linewidths'])]

    def _draw_polygons(self, axes: Axes, node: NDArray, indices: NDArray, kwargs) -> list[Collection]:
        if indices.size == 0:
            return []
        return [A.poly(axes=axes, points=node, struct=indices[kwargs['index']],
                       edgecolor=kwargs['edgecolor'], cellcolor=kwargs['cellcolor'],
                       linewidths=kwargs['linewidths'], alpha=kwargs['alpha'])]

    def _draw_surface(self, axes: Axes, node: NDArray, sector, kwargs) -> list[Collection]:
        collections: list[Collection] = []
        view = self.mesh.Entity(sector.schema_name)
        index = kwargs.get('index', slice(None))
        kwargs['index'] = slice(None)

        for face_name in sector.schema.OFace.keys():
            face_view = self.mesh.Entity(face_name)
            relation = view.to(face_view).tgt_indices[index]
            face_index = np.unique(np.asarray(relation).reshape(-1))
            face_index = face_index[np.asarray(face_view.boundary().mask)[face_index]]
            indices = np.asarray(face_view.indices)[face_index]
            if indices.ndim == 1:
                indices = indices.reshape(1, -1)
            collections.extend(self._draw_polygons(axes, node, indices, kwargs))
        return collections

    def _draw_sector(self, axes: Axes, node: NDArray, sector, kwargs) -> list[Collection]:
        indices = np.asarray(sector.indices)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)

        if sector.schema.top_dim == 0:
            return self._draw_nodes(axes, node, kwargs)
        if sector.schema.top_dim == 1:
            return self._draw_edges(axes, node, indices, kwargs)
        if sector.schema.top_dim == 2:
            ccw = sector.schema.ccw
            if self.mesh.top_dimension() == 2 and ccw is not None:
                indices = indices[:, ccw]
            return self._draw_polygons(axes, node, indices, kwargs)
        if sector.schema.top_dim == 3:
            return self._draw_surface(axes, node, sector, kwargs)
        return []

    def draw(self, axes: Axes, *args, **kwargs):
        """Draw mesh entities selected by topological dimension or entity name.

        With no selection, draw only the highest topological dimension. If that
        dimension is 3, draw only the boundary surface. Users may pass an entity
        name such as ``'tri'``/``'tet'``, an alias such as ``'cell'``/``'face'``,
        a dimension such as ``2``, or a list through ``entities``.
        """
        self.set_lim(axes, kwargs['box'])
        axes.set_aspect(kwargs['aspect'])
        self.set_show_axis(axes, kwargs['showaxis'])

        node = self._node_array()
        collections: list[Collection] = []

        from ..schema.registry import schema_name_multi_parser
        entity = kwargs.get('entity')
        assert isinstance(entity, (str, int))
        names = schema_name_multi_parser(
            entity,
            self.mesh.top_dimension(),
            self.mesh.block.sectors.keys()
        )
        # surface_drawn = False
        for name in names:
            sector = self.mesh.block.get_sector(name)
            collections.extend(self._draw_sector(axes, node, sector, kwargs))

        return collections


class EntityFinder(MeshPloter):
    def __init__(self, mesh: Mesh) -> None:
        super().__init__(mesh)
        self._args.update(
            color = 'r',
            etype = 'cell',
            showindex = False,
            multiindex = None,
            fontcolor = 'k',
            fontsize = 24,
        )

    def draw(self, axes: Axes, *args, **kwargs):
        """Show the barycenter of selected entities."""
        etype_or_node = kwargs['etype']
        color = kwargs['color']

        if isinstance(etype_or_node, str):
            if etype_or_node in _ENTITY_NAMES:
                bc = np.asarray(self.mesh.Entity(etype_or_node).barycenter(index=kwargs['index']))
            else:
                bcs = [np.asarray(view.barycenter(index=kwargs['index']))
                       for view in self.mesh.Entities(etype_or_node)]
                if not bcs:
                    raise ValueError(f"No entity found for {etype_or_node!r}.")
                bc = np.concatenate(bcs, axis=0) if len(bcs) > 1 else bcs[0]
        elif isinstance(etype_or_node, int):
            bcs = [np.asarray(view.barycenter(index=kwargs['index']))
                   for view in self.mesh.Entities(etype_or_node)]
            if not bcs:
                raise ValueError(f"No entity found for dimension {etype_or_node!r}.")
            bc = np.concatenate(bcs, axis=0) if len(bcs) > 1 else bcs[0]
        elif isinstance(etype_or_node, np.ndarray):
            bc = etype_or_node
        else:
            raise TypeError(f"Invalid entity type or node info.")

        bc = np.asarray(bc)

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
