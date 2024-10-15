
from typing import (
    Union, Optional, Dict, Sequence, overload, Callable, Tuple, Any
)

import numpy as np
import torch

from ..typing import (
    Tensor, Index, EntityName, _S,
    _int_func, _dtype, _device
)
from .. import logger
from . import functional as F
from .utils import estr2dim, edim2entity, edim2node, mesh_top_csr, MeshMeta
from .quadrature import Quadrature
import torch
from scipy.sparse import csr_matrix


##################################################
### Mesh Data Structure Base
##################################################
# NOTE: MeshDS provides a storage for mesh entities and all topological methods.

class MeshDS(metaclass=MeshMeta):
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
        assert hasattr(self, '_entity_dim_method_name_map')
        self._entity_storage: Dict[int, Tensor] = {}
        self._entity_factory: Dict[int, Callable] = {
            k: getattr(self, self._entity_dim_method_name_map[k])
            for k in self._entity_dim_method_name_map
        }
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> Tensor: ...
    def __getattr__(self, name: str):
        if name in self._STORAGE_ATTR:
            etype_dim = estr2dim(self, name)
            return edim2entity(self._entity_storage, self._entity_factory, etype_dim)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = estr2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._STORAGE_ATTR:
            del self._entity_storage[estr2dim(self, name)]
        else:
            super().__delattr__(name)

    ### cuda
    def to(self, device: Union[_device, str, None]=None, non_blocking=False):
        for edim in self._entity_storage.keys():
            entity = self._entity_storage[edim]
            self._entity_storage[edim] = entity.to(device, non_blocking=non_blocking)
        for attr in self.__dict__:
            value = self.__dict__[attr]
            if isinstance(value, Tensor):
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
        entity = self.entity(etype)

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
            index (int | slice | Tensor): The index of the entity.\n
            etype (int | str): The topological dimension of the entity, or name\
            'cell' | 'face' | 'edge' | 'node'.\n
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: Entity or the default value. Returns None if not found.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        return edim2entity(self.storage(), self._entity_factory, etype, index)

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
        
    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[
            face2cell[:, [0]],
            localFace2edge[face2cell[:, 2]]
        ]
        if return_sparse is False:
            return face2edge
        else:
            NF = self.number_of_faces()
            NE = self.number_of_edges()
            NEF = self.number_of_edges_of_faces()
            f2e = csr_matrix(
                (
                    torch.ones(NEF*NF, dtype=torch.bool),
                    (
                        torch.repeat(torch.arange(NF), NEF),
                        face2edge.view(-1)
                    )
                ), shape=(NF, NE))
            return f2e
    
    def cell_to_face(self, index: Index=_S, *, dtype: Optional[_dtype]=None, return_sparse=False) -> Tensor:
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        NFC = self.number_of_faces_of_cells()

        face2cell = self.face2cell
        dtype = dtype if dtype is not None else self.itype

        if not torch.is_floating_point(torch.tensor(0, dtype=dtype)):
            dtype = torch.int64

        cell2face = torch.zeros((NC, NFC), dtype=dtype)
        arange_tensor = torch.arange(NF, dtype=dtype)

        assert cell2face.dtype == arange_tensor.dtype, f"Data type mismatch: cell2face is {cell2face.dtype}, arange_tensor is {arange_tensor.dtype}"

        cell2face[face2cell[:, 0], face2cell[:, 2]] = arange_tensor
        cell2face[face2cell[:, 1], face2cell[:, 3]] = arange_tensor
        if not return_sparse:
            return cell2face
        else:
            return mesh_top_csr(cell2face[index], self.number_of_faces(), dtype=dtype)


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
    pass

    # NOTE: Here are some examples for entity factories:
    # implement them in subclasses if necessary.

    # @entitymethod
    # def _node(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _edge(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _face(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _cell(self, index: Index=_S):
    #     raise NotImplementedError
