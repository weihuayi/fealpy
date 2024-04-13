
from typing import Union, TypeVar, Generic, Dict, Sequence, overload
import torch

from .quadrature import Quadrature

Tensor = torch.Tensor
Index = Union[Tensor, int, slice]
Entity = Union[Tensor, Sequence[Tensor]]
_S = slice(None, None, None)


class MeshDataStructureBase():
    def __init__(self) -> None:
        self._entity_storage: Dict[int, Entity] = {}
        self._initialized = True

    def __getattr__(self, name: str):
        if name not in {'cell', 'face', 'edge'}:
            return super().__getattr__(name)
        etype_dim = self._entity_str2dim(name)
        return self._dim2entity(etype_dim)

    def __setattr__(self, name: str, value: torch.Any) -> None:
        if not hasattr(self, '_initialized'):
            raise RuntimeError('please call super().__init__() before setting attributes.')
        if name in ('cell', 'face', 'edge'):
            etype_dim = self._entity_str2dim(name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    def top_dimension(self) -> int:
        raise NotImplementedError

    def geo_dimension(self) -> int:
        raise NotImplementedError

    GD = property(geo_dimension)
    TD = property(top_dimension)

    def construct(self) -> None:
        raise NotImplementedError

    # Get the entity's top dimension from its name.
    def _entity_str2dim(self, etype: str) -> int:
        if etype == 'cell':
            return self.top_dimension()
        elif etype == 'face':
            TD = self.top_dimension()
            if TD <= 1:
                raise ValueError('the mesh has no face.')
            return TD - 1
        elif etype == 'edge':
            return 1
        elif etype == 'node':
            raise ValueError('the node is not in mesh structure.')
        else:
            raise ValueError(f'{etype} is not a valid entity type.')

    # Get the entity from its toppology dimension.
    def _dim2entity(self, etype_dim: int, index: Index=_S) -> Entity:
        if etype_dim in self._entity_storage:
            return self._entity_storage[etype_dim][index]
        else:
            raise ValueError(f'{etype_dim} is not a valid entity dimension.')

    def entity(self, etype: Union[int, str], index: Index=_S) -> Entity:
        r"""@brief Get entities in mesh structure.

        @param etype: int or str. The topology dimension of the entity, or name
        'cell' | 'face' | 'edge'. Note that 'node' is not in mesh structure.
        @param index: int, slice ot Tensor. The index of the entity.

        @return: Tensor or Sequence[Tensor].
        """
        if isinstance(etype, str):
            etype = self._entity_str2dim(etype)
        return self._dim2entity(etype, index)

    def total_face(self) -> Entity:
        raise NotImplementedError

    def total_edge(self) -> Entity:
        raise NotImplementedError

    def number_of_nodes(self) -> int:
        return len(self._dim2entity(0))

    def number_of_edges(self) -> int:
        return len(self._dim2entity(1))

    def number_of_faces(self) -> int:
        return len(self._dim2entity(self.top_dimension() - 1))

    def number_of_cells(self) -> int:
        return len(self._dim2entity(self.top_dimension()))


_MDS_co = TypeVar('_MDS_co', bound=MeshDataStructureBase, covariant=True)


class MeshBase(Generic[_MDS_co]):
    ds: _MDS_co
    node: Tensor

    def geo_dimension(self) -> int:
        return self.ds.geo_dimension()

    def top_dimension(self) -> int:
        return self.ds.top_dimension()

    GD = property(geo_dimension)
    TD = property(top_dimension)

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_nodes(self) -> int:
        return self.ds.number_of_nodes()

    def entity(self, etype: Union[int, str], index: Index=_S) -> Entity:
        if etype in ('node', 0):
            return self.node[index]
        else:
            return self.ds.entity(etype, index)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        r"""@brief Get the quadrature points and weights."""
        raise NotImplementedError


class HomoMeshBase(MeshBase[_MDS_co]):
    @overload
    def entity(self, etype: Union[int, str], index: Index=_S) -> Tensor: ...
    def entity_barycenter(self, etype: Union[int, str], index: Index=_S) -> Tensor:
        r"""@brief Get the barycenter of the entity.

        @param etype: int or str. The topology dimension of the entity, or name
        'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
        @param index: int, slice ot Tensor. The index of the entity.

        @return: Tensor.
        """
        node = self.entity('node')
        if etype in ('node', 0):
            return node[index]
        else:
            entity: Tensor = self.ds.entity(etype, index)
            return torch.mean(node[entity, :], dim=1)

    def bc_to_point(self, bcs: Union[Tensor, Sequence[Tensor]], index=_S) -> Tensor:
        r"""@brief Convert barycenter coordinate points to cartesian coordinate points
            on mesh entities.
        """
        node = self.entity('node')
        if isinstance(bcs, Tensor): # for edge, interval, triangle and tetrahedron
            TD = bcs.size(-1)
            entity: Tensor = self.ds.entity(TD, index)
            return torch.einsum('ijk, ...j -> ...ik', node[entity, :], bcs)

        else: # for quadrangle and hexahedron
            TD = len(bcs)
            entity: Tensor = self.ds.entity(TD, index)
            desp1 = 'mnopq'
            desp2 = 'abcde'
            string = ", ".join([desp1[i]+desp2[i] for i in range(TD)])
            string += " -> " + desp1[:TD] + desp2[:TD]
            return torch.einsum(string, *bcs).reshape(-1, entity.size(-1))
