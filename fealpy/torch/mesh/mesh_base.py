
from typing import (
    Union, Optional, Dict, Sequence, overload, Callable, Literal, Tuple
)

import numpy as np
import torch

from .. import logger
from . import functional as F
from .quadrature import Quadrature

Tensor = torch.Tensor
Index = Union[Tensor, int, slice]
EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
_int_func = Callable[..., int]
_dtype = torch.dtype
_device = torch.device

_S = slice(None, None, None)


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


def estr2dim(mesh, estr: str) -> int:
    if estr == 'cell':
        return mesh.top_dimension()
    elif estr == 'face':
        TD = mesh.top_dimension()
        return TD - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity name in FEALPy.')


def edim2entity(dict_: Dict, edim: int, index=None):
    r"""Get entity tensor by its top dimension. Returns None if not found."""
    if edim in dict_:
        et = dict_[edim]
        if index is None:
            return et
        else: # TODO: finish this for homogeneous mesh
            return et[index]
    else:
        logger.info(f'entity {edim} is not found and a NoneType is returned.')
        return None


def edim2node(mesh, etype_dim: int, index=None, dtype=None) -> Tensor:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = edim2entity(mesh.storage(), etype_dim, index)
    location = getattr(entity, 'location', None)
    NN = mesh.count('node')
    if NN <= 0:
        raise RuntimeError('No valid node is found in the mesh.')
    return mesh_top_csr(entity, NN, location, dtype=dtype)


##################################################
### Mesh Data Structure Base
##################################################
# NOTE: MeshDS provides a storage for mesh entities and all topological methods.

class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: Tensor
    face: Tensor
    edge: Tensor
    node: Tensor
    face2cell: Tensor
    cell2edge: Tensor
    localEdge: Tensor # only for homogeneous mesh
    localFace: Tensor # only for homogeneous mesh

    def __init__(self, TD: int) -> None:
        self._entity_storage: Dict[int, Tensor] = {}
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Tensor: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return object.__getattribute__(self, name)
        etype_dim = estr2dim(self, name)
        return edim2entity(self.storage(), etype_dim)

    def __setattr__(self, name: str, value: torch.Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = estr2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    ### cuda
    def to(self, device: Union[_device, str, None]=None, non_blocking=False):
        for edim in self._entity_storage.keys():
            entity = self._entity_storage[edim]
            self._entity_storage[edim] = entity.to(device, non_blocking=non_blocking)
        for attr in self.__dict__:
            value = self.__dict__[attr]
            if isinstance(value, torch.Tensor):
                self.__dict__[attr] = value.to(device, non_blocking=non_blocking)
        return self

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> _dtype: return self.cell.dtype
    @property
    def device(self) -> _device: return self.cell.device
    def storage(self) -> Dict[int, Tensor]:
        return self._entity_storage

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        if isinstance(etype, str):
            edim = estr2dim(self, etype)
        entity = edim2entity(self.storage(), edim)

        if entity is None:
            logger.info(f'count: entity {etype} is not found and 0 is returned.')
            return 0

        if hasattr(entity, 'location'):
            return entity.location.size(0) - 1
        else:
            return entity.size(0)

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def _nv_entity(self, etype: Union[int, str]) -> Tensor:
        entity = self.entity(etype)
        if hasattr(entity, 'location'):
            loc = entity.location
            return loc[1:] - loc[:-1]
        else:
            return torch.tensor((entity.shape[-1],), dtype=self.itype, device=self.device)

    def number_of_vertices_of_cells(self): return self._nv_entity('cell')
    def number_of_vertices_of_faces(self): return self._nv_entity('face')
    def number_of_vertices_of_edges(self): return self._nv_entity('edge')
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        """Get entities in mesh structure.

        Parameters:
            index (int | slice | Tensor): The index of the entity.

            etype (int | str): The topological dimension of the entity, or name
            'cell' | 'face' | 'edge' | 'node'.

            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: Entity or the default value. Returns None if not found.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        return edim2entity(self.storage(), etype, index)

    ### topology
    def cell_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension()
        return edim2node(self, etype, index, dtype=dtype)

    def face_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        etype = self.top_dimension() - 1
        return edim2node(self, etype, index, dtype=dtype)

    def edge_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> Tensor:
        return edim2node(self, 1, index, dtype)

    def cell_to_edge(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> Tensor:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge[index]
        if return_sparse:
            return mesh_top_csr(cell2edge[index], self.number_of_edges(), dtype=dtype)
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
    def boundary_node_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = torch.zeros((NN,), **kwargs)
        bd_node_flag[bd_face2node.ravel()] = True
        return bd_node_flag

    def boundary_face_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> Tensor:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            Tensor: boundary cell flag.
        """
        NC = self.number_of_cells()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = torch.zeros((NC,), **kwargs)
        bd_cell_flag[bd_face2cell.ravel()] = True
        return bd_cell_flag

    def boundary_node_index(self): return self.boundary_node_flag().nonzero().ravel()
    # TODO: finish this:
    # def boundary_edge_index(self): return self.boundary_edge_flag().nonzero().ravel()
    def boundary_face_index(self): return self.boundary_face_flag().nonzero().ravel()
    def boundary_cell_index(self): return self.boundary_cell_flag().nonzero().ravel()

    ### Homogeneous Mesh ###
    def is_homogeneous(self, etype: Union[int, str]='cell') -> bool:
        """Return True if the mesh entity is homogeneous.

        Returns:
            bool: Homogeneous indicator.
        """
        entity = self.entity(etype)
        if entity is None:
            raise RuntimeError(f'{etype} is not found.')
        return entity.ndim == 2

    def total_face(self) -> Tensor:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> Tensor:
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        NVE = local_edge.shape[-1]
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalFace = self.total_face()
        _, i0_np, j_np = np.unique(
            torch.sort(totalFace, dim=1)[0].cpu().numpy(),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = np.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = np.arange(NFC*NC, dtype=i0_np.dtype)

        self.cell2face = torch.from_numpy(j_np).to(self.device).reshape(NC, NFC)

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = torch.from_numpy(face2cell_np).to(self.device)

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = np.unique(
                torch.sort(total_edge, dim=1)[0].cpu().numpy(),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = torch.from_numpy(j).to(self.device).reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell
            self.cell2edge = self.cell2face

        logger.info(f"Mesh toplogy relation constructed, with {NF} faces, "
                    f"on device {self.device}")


##################################################
### Mesh Base
##################################################

class Mesh(MeshDS):
    @property
    def ftype(self) -> _dtype:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the float type as the node '
                               'has not been assigned.')
        return node.dtype

    def geo_dimension(self) -> int:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, etype: int) -> Tensor:
        return F.multi_index_matrix(p, etype, dtype=self.itype, device=self.device)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        """Get the barycenter of the entity.

        Parameters:
            etype (int | str): The topology dimension of the entity, or name
                'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: A 2-d tensor containing barycenters of the entity.
        """
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]

        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        etn = edim2node(self, etype, index, dtype=node.dtype)
        return F.entity_barycenter(etn, node)

    def edge_length(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the length of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor[NE,]: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return F.edge_length(edge, self.node, out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> Tensor:
        """Calculate the normal of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.\n
            unit (bool, optional): _description_. Defaults to False.\n
            out (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor[NE, GD]: _description_
        """
        edge = self.entity(1, index=index)
        return F.edge_normal(edge, self.node, unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> Tensor:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, unit=True, out=out)

    def quadrature_formula(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights.

        Parameters:
            q (int): The index of the quadrature points.
            etype (int | str, optional): The topology dimension of the entity to\
            generate the quadrature points on. Defaults to 'cell'.

        Returns:
            Quadrature: Object for quadrature points and weights.
        """
        raise NotImplementedError

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        logger.warning("The `integrator` is deprecated and will be removed after 3.0. "
                       "Use `quadrature_formula` instead.")
        return self.quadrature_formula(q, etype, qtype)

    # ipoints
    def edge_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        kwargs = {'dtype': edges.dtype, 'device': self.device}
        indices = torch.arange(NE, **kwargs)[index]
        return torch.cat([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + torch.arange(p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], dim=-1)

    # shape function
    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        """Shape function value on the given bc points, in shape (..., ldof).

        Parameters:
            bc (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variable (str, optional): The variable name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof). The shape will\
            be (1, NQ, ldof) if `variable == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        """Gradient of shape function on the given bc points, in shape (..., ldof, bc).

        Parameters:
            bc (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variable (str, optional): The variable name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof, bc). The shape will\
            be (NC, NQ, ldof, GD) if `variable == 'x'`.
        """
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomogeneousMesh(Mesh):
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> Tensor:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return F.homo_entity_barycenter(entity, node)

    def bc_to_point(self, bcs: Union[Tensor, Sequence[Tensor]],
                    etype: Union[int, str]='cell', index: Index=_S) -> Tensor:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
        """
        node = self.entity('node')
        entity = self.entity(etype, index)
        order = getattr(entity, 'bc_order', None)
        return F.bc_to_points(bcs, node, entity, order)

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> Tensor:
        raise NotImplementedError


class SimplexMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    # shape function
    def grad_lambda(self, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.itype, device=self.device)
        phi = F.simplex_shape_function(bc, p, mi)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return phi.unsqueeze_(0)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

    def grad_shape_function(self, bc: Tensor, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.itype, device=self.device)
        R = F.simplex_grad_shape_function(bc, p, mi) # (NQ, ldof, bc)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = torch.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'q': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")


class TensorMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return F.tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int) -> int:
        return F.tensor_gdof(p, self)

    # shape function
    def grad_lambda(self, index: Index=_S) -> Tensor:
        raise NotImplementedError

    def shape_function(self, bc: Tuple[Tensor], p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        pass

    def grad_shape_function(self, bc: Tuple[Tensor], p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[Tensor]=None) -> Tensor:
        pass


class StructuredMesh(HomogeneousMesh):
    _STORAGE_METH = ['_node', '_edge', '_face', '_cell']

    @overload
    def __getattr__(self, name: EntityName) -> Tensor: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return object.__getattribute__(self, name)
        etype_dim = estr2dim(self, name)

        if etype_dim in self._entity_storage:
            return self._entity_storage[etype_dim]
        else:
            _method = self._STORAGE_METH[etype_dim]

            if not hasattr(self, _method):
                raise AttributeError(
                    f"'{name}' in structured mesh requires a factory method "
                    f"'{_method}' to generate the entity data when {name} "
                    "is not in the storage."
                )

            entity = getattr(self, _method)()
            self._entity_storage[etype_dim] = entity

            return entity

    def _node(self):
        raise NotImplementedError

    def _edge(self):
        raise NotImplementedError

    def _face(self):
        raise NotImplementedError

    def _cell(self):
        raise NotImplementedError
