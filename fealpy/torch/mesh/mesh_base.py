
from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import torch

from . import functional as F
from .quadrature import Quadrature

Tensor = torch.Tensor
Index = Union[Tensor, int, slice]
EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
_int_func = Callable[..., int]
_dtype = torch.dtype
_device = torch.device

_S = slice(None, None, None)
_T = TypeVar('_T')
_default = object()


##################################################
### Utils
##################################################

def mesh_top_csr(entity: Tensor, num_targets: int, location: Optional[Tensor]=None, *,
                 dtype: Optional[_dtype]=None) -> Tensor:
    r"""CSR format of a mesh topology relaionship matrix."""
    device = entity.device

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = torch.arange(
            entity.size(0) + 1, dtype=entity.dtype, device=device
        ).mul_(entity.size(1))
    else:
        raise ValueError('dimension of entity must be 1 or 2.')

    return torch.sparse_csr_tensor(
        crow,
        entity.reshape(-1),
        torch.ones(entity.numel(), dtype=dtype, device=device),
        size=(entity.size(0), num_targets),
        dtype=dtype, device=device
    )


def entity_str2dim(ds, etype: str) -> int:
    if etype == 'cell':
        return ds.top_dimension()
    elif etype == 'cell_location':
        return -ds.top_dimension()
    elif etype == 'face':
        TD = ds.top_dimension()
        if TD <= 1:
            raise ValueError('the mesh has no face entity.')
        return TD - 1
    elif etype == 'face_location':
        TD = ds.top_dimension()
        if TD <= 1:
            raise ValueError('the mesh has no face location.')
        return -TD + 1
    elif etype == 'edge':
        return 1
    elif etype == 'node':
        return 0
    else:
        raise KeyError(f'{etype} is not a valid entity attribute.')


def entity_dim2tensor(ds, etype_dim: int, index=None, *, default=_default):
    r"""Get entity tensor by its top dimension."""
    if etype_dim in ds._entity_storage:
        et = ds._entity_storage[etype_dim]
        if index is None:
            return et
        else:
            if et.ndim == 1:
                raise RuntimeError("index is not supported for flattened entity.")
            return et[index]
    else:
        if default is not _default:
            return default
        raise ValueError(f'{etype_dim} is not a valid entity attribute index '
                         f"in {ds.__class__.__name__}.")


def entity_dim2node(ds, etype_dim: int, index=None, dtype=None) -> Tensor:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = entity_dim2tensor(ds, etype_dim, index)
    location = entity_dim2tensor(ds, -etype_dim, default=None)
    return mesh_top_csr(entity, ds.number_of_nodes(), location, dtype=dtype)


##################################################
### Mesh Data Structure Base
##################################################

class MeshDataStructure():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'cell_location','face_location']
    def __init__(self, NN: int, TD: int) -> None:
        self._entity_storage: Dict[int, Tensor] = {}
        self.NN = NN
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Tensor: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return self.__dict__[name]
        etype_dim = entity_str2dim(self, name)
        return entity_dim2tensor(self, etype_dim)

    def __setattr__(self, name: str, value: torch.Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = entity_str2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    ### cuda
    def to(self, device: Union[_device, str, None]=None, non_blocking=False):
        for entity_tensor in self._entity_storage.values():
            entity_tensor.to(device, non_blocking=non_blocking)
        return self

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> _dtype: return self.cell.dtype
    @property
    def device(self) -> _device: return self.cell.device

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """@brief Return the number of entities of the given type."""
        if etype in ('node', 0):
            return self.NN
        if isinstance(etype, str):
            edim = entity_str2dim(self, etype)
        if -edim in self._entity_storage: # for polygon mesh
            return self._entity_storage[-edim].size(0) - 1
        return entity_dim2tensor(self, edim).size(0) # for homogeneous mesh

    def number_of_nodes(self): return self.NN
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    ### constructors
    def construct(self) -> None:
        raise NotImplementedError

    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor: ...
    @overload
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default: _T) -> Union[Tensor, _T]: ...
    def entity(self, etype: Union[int, str], index: Optional[Index]=None, *, default=_default):
        r"""@brief Get entities in mesh structure.

        @param etype: int or str. The topology dimension of the entity, or name
        'cell' | 'face' | 'edge'. Note that 'node' is not in mesh structure.
        For polygon meshes, the names 'cell_location' | 'face_location' may also be
        available, and the `index` argument is applied on the flattened entity tensor.
        @param index: int, slice ot Tensor. The index of the entity.

        @return: Tensor or Sequence[Tensor].
        """
        if isinstance(etype, str):
            etype = entity_str2dim(self, etype)
        return entity_dim2tensor(self, etype, index, default=default)

    def total_face(self) -> Tensor:
        raise NotImplementedError

    def total_edge(self) -> Tensor:
        raise NotImplementedError

    ### topology
    def cell_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension()
        return entity_dim2node(self, etype, index, dtype=dtype)

    def face_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension() - 1
        return entity_dim2node(self, etype, index, dtype=dtype)

    def edge_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        return entity_dim2node(self, 1, index, dtype)

    def cell_to_edge(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> Tensor:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge[index]
        if return_sparse:
            return mesh_top_csr(cell2edge[index, :2], self.number_of_edges(), dtype=dtype)
        else:
            return cell2edge[index]

    def face_to_cell(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> Tensor:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        if return_sparse:
            return mesh_top_csr(face2cell[index, :2], self.number_of_cells(), dtype=dtype)
        else:
            return face2cell[index]

    ### boundary
    def boundary_face_flag(self): return self.face2cell[:, 0] == self.face2cell[:, 1]
    def boundary_face_index(self): return torch.nonzero(self.boundary_face_flag(), as_tuple=True)[0]


class HomoMeshDataStructure(MeshDataStructure):
    ccw: Tensor
    localEdge: Tensor
    localFace: Tensor

    def __init__(self, NN: int, TD: int, cell: Tensor) -> None:
        super().__init__(NN, TD)
        self.cell = cell

    number_of_vertices_of_cells: _int_func = lambda self: self.cell.shape[-1]
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]
    number_of_vertices_of_faces: _int_func = lambda self: self.localFace.shape[-1]
    number_of_vertices_of_edges: _int_func = lambda self: self.localEdge.shape[-1]

    def total_face(self) -> Tensor:
        NVF = self.number_of_faces_of_cells()
        cell = self.entity(self.TD)
        local_face = self.localFace
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> Tensor:
        NVE = self.number_of_vertices_of_edges()
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self) -> None:
        raise NotImplementedError


##################################################
### Mesh Base
##################################################

class Mesh():
    ds: MeshDataStructure
    node: Tensor

    def to(self, device: Union[_device, str, None]=None, non_blocking: bool=False):
        self.ds.to(device, non_blocking)
        self.node = self.node.to(device, non_blocking)
        return self

    @property
    def ftype(self) -> _dtype: return self.node.dtype
    @property
    def device(self) -> _device: return self.node.device
    def geo_dimension(self) -> int: return self.node.shape[-1]
    def top_dimension(self) -> int: return self.ds.top_dimension()
    GD = property(geo_dimension)
    TD = property(top_dimension)

    def multi_index_matrix(self, p: int, etype: int) -> Tensor:
        return F.multi_index_matrix(p, etype, dtype=self.ds.itype, device=self.device)

    def count(self, etype: Union[int, str]) -> int: return self.ds.count(etype)
    def number_of_cells(self) -> int: return self.ds.number_of_cells()
    def number_of_faces(self) -> int: return self.ds.number_of_faces()
    def number_of_edges(self) -> int: return self.ds.number_of_edges()
    def number_of_nodes(self) -> int: return self.ds.number_of_nodes()
    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]
        else:
            return self.ds.entity(etype, index)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        r"""@brief Get the barycenter of the entity.

        @param etype: int or str. The topology dimension of the entity, or name
        'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
        @param index: int, slice ot Tensor. The index of the entity.

        @return: Tensor.
        """
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]

        node = self.node
        if isinstance(etype, str):
            etype = entity_str2dim(self.ds, etype)
        etn = entity_dim2node(self.ds, etype, index, dtype=node.dtype)
        return F.entity_barycenter(etn, node)

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        r"""@brief Get the quadrature points and weights."""
        raise NotImplementedError

    def shape_function(self, bc: Tensor, p: int=1, *, index: Tensor,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        """@brief Shape function value on the given bc points, in shape (..., ldof).

        @param bc: The bc points, in shape (..., NVC).
        @param p: The order of the shape function.
        @param index: The index of the cell.
        @param variable: The variable name.
        @param mi: The multi-index matrix.

        @returns: The shape function value, in shape (..., ldof).
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Tensor,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: Tensor, p: int=1, *, index: Tensor,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomoMesh(Mesh):
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.ds.entity(etype, index)
        return F.homo_entity_barycenter(entity, node)

    def bc_to_point(self, bcs: Union[Tensor, Sequence[Tensor]], etype='cell', index=_S) -> Tensor:
        r"""@brief Convert barycenter coordinate points to cartesian coordinate points
            on mesh entities.
        """
        node = self.entity('node')
        entity = self.ds.entity(etype, index)
        # TODO: finish this
        # ccw = getattr(self.ds, 'ccw', None)
        ccw = None
        return F.bc_to_points(bcs, node, entity, ccw)

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def edge_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        r"""@brief Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.ds.edge[index]
        kwargs = {'dtype': edges.dtype, 'device': self.device}
        indices = torch.arange(NE, **kwargs)[index]
        return torch.cat([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + torch.arange(p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], dim=-1)

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError
