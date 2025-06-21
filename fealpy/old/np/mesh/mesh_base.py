from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .. import logger

from .utils import EntityName, Index, _T, _S, _int_func, _dtype
from .utils import estr2dim, edim2entity, edim2node, mesh_top_csr, arr_to_csr
from . import functional as F
from .quadrature import Quadrature


class MeshDS():
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: NDArray
    face: NDArray
    edge: NDArray
    node: NDArray
    face2cell: NDArray
    cell2edge: NDArray
    localEdge: NDArray # only for homogeneous mesh
    localFace: NDArray # only for homogeneous mesh

    def __init__(self, TD: int) -> None:
        self._entity_storage: Dict[int, NDArray] = {}
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> NDArray: ...
    def __getattr__(self, name: str):
        if name not in self._STORAGE_ATTR:
            return object.__getattribute__(self, name)
        etype_dim = estr2dim(self, name)
        return edim2entity(self.storage(), etype_dim)

    def __setattr__(self, name: str, value: np.any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = estr2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> _dtype: return self.cell.dtype
    def storage(self) -> Dict[int, NDArray]:
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
            return entity.location.shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def _nv_entity(self, etype: Union[int, str]) -> NDArray:
        entity = self.entity(etype)
        if hasattr(entity, 'location'):
            loc = entity.location
            return loc[1:] - loc[:-1]
        else:
            return np.array((entity.shape[-1],), dtype=self.itype)

    def number_of_vertices_of_cells(self): return self._nv_entity('cell')
    def number_of_vertices_of_faces(self): return self._nv_entity('face')
    def number_of_vertices_of_edges(self): return self._nv_entity('edge')
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> NDArray:
        """Get entities in mesh structure.

        Parameters:
            index (int | slice | NDArray): The index of the entity.

            etype (int | str): The topological dimension of the entity, or name
            'cell' | 'face' | 'edge' | 'node'.

            index (int | slice | NDArray): The index of the entity.

        Returns:
            NDArray: Entity or the default value. Returns None if not found.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        return edim2entity(self.storage(), etype, index)

    ### topology
    def cell_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> NDArray:
        etype = self.top_dimension()
        return edim2node(self, etype, index, dtype=dtype)

    def face_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> NDArray:
        etype = self.top_dimension() - 1
        return edim2node(self, etype, index, dtype=dtype)

    def edge_to_node(self, index: Optional[Index]=None, *, dtype: Optional[_dtype]=None) -> NDArray:
        return edim2node(self, 1, index, dtype)

    def cell_to_edge(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> NDArray:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        cell2edge = self.cell2edge[index]
        if return_sparse:
            return mesh_top_csr(cell2edge[index], self.number_of_edges(), dtype=dtype)
        else:
            return cell2edge[index]

    def face_to_cell(self, index: Index=_S, *, dtype: Optional[_dtype]=None,
                     return_sparse=False) -> NDArray:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        if return_sparse:
            return mesh_top_csr(face2cell[index, :2], self.number_of_cells(), dtype=dtype)
        else:
            return face2cell[index]

    ### boundary
    def boundary_node_flag(self) -> NDArray:
        """Return a boolean NDArray indicating the boundary nodes.

        Returns:
            NDArray: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype}
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = np.zeros((NN,), **kwargs)
        bd_node_flag[bd_face2node.ravel()] = True
        return bd_node_flag

    def boundary_face_flag(self) -> NDArray:
        """Return a boolean NDArray indicating the boundary faces.

        Returns:
            NDArray: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> NDArray:
        """Return a boolean NDArray indicating the boundary cells.

        Returns:
            NDArray: boundary cell flag.
        """
        NC = self.number_of_cells()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = np.zeros((NC,), **kwargs)
        bd_cell_flag[bd_face2cell.ravel()] = True
        return bd_cell_flag

    def boundary_node_index(self): return self.boundary_node_flag().nonzero()[0]
    # TODO: finish this:
    # def boundary_edge_index(self): return self.boundary_edge_flag().nonzero().ravel()
    def boundary_face_index(self): return self.boundary_face_flag().nonzero()[0]
    def boundary_cell_index(self): return self.boundary_cell_flag().nonzero()[0]

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

    def total_face(self) -> NDArray:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> NDArray:
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        NVE = local_edge.shape[-1]
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.cell.shape[0]
        NFC = self.cell.shape[1]

        totalFace = self.total_face()
        sorted_face = np.sort(totalFace, axis=1)
        _, i0_np, j_np = np.unique(
            sorted_face,
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = np.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = np.arange(NFC*NC, dtype=i0_np.dtype)

        self.cell2edge = j_np.reshape(NC, NFC)
        self.cell2face = self.cell2edge

        face2cell_np = np.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = face2cell_np
        self.edge2cell = self.face2cell

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = np.unique(
                np.sort(total_edge, dim=1)[0],
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = j.reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell

        logger.info(f"Mesh toplogy relation constructed, with {NF} faces, ")


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

    def multi_index_matrix(self, p: int, etype: int) -> NDArray:
        return F.multi_index_matrix(p, etype, dtype=self.itype)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> NDArray:
        """Get the barycenter of the entity.

        Args:
            etype (int | str): The topology dimension of the entity, or name
            'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | NDArray): The index of the entity.

        Returns:
            NDArray: A 2-d NDArray containing barycenters of the entity.
        """
        if etype in ('node', 0):
            return self.node if index is None else self.node[index]

        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        etn = edim2node(self, etype, index, dtype=node.dtype)
        return F.entity_barycenter(etn, node)

    def edge_length(self, index: Index=_S, out=None) -> NDArray:
        """Calculate the length of the edges.

        Args:
            index (int | slice | NDArray, optional): Index of edges.
            out (NDArray, optional): The output NDArray. Defaults to None.

        Returns:
            NDArray: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return F.edge_length(self.node[edge], out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> NDArray:
        """Calculate the normal of the edges.

        Args:
            index (int | slice | NDArray, optional): Index of edges.
            unit (bool, optional): _description_. Defaults to False.
            out (NDArray, optional): _description_. Defaults to None.

        Returns:
            NDArray: _description_
        """
        edge = self.entity(1, index=index)
        return F.edge_normal(self.node[edge], unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> NDArray:
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

    def shape_function(self, bc: NDArray, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[NDArray]=None) -> NDArray:
        """Shape function value on the given bc points, in shape (..., ldof).

        Args:
            bc (NDArray): The bc points, in shape (..., NVC).
            p (int, optional): The order of the shape function. Defaults to 1.
            index (int | slice | NDArray, optional): The index of the cell.
            variable (str, optional): The variable name. Defaults to 'u'.
            mi (NDArray, optional): The multi-index matrix. Defaults to None.

        Returns:
            NDArray: The shape function value with shape (..., ldof). The shape will\
            be (..., 1, ldof) if `variable == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: NDArray, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[NDArray]=None) -> NDArray:
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: NDArray, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[NDArray]=None) -> NDArray:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")


class HomogeneousMesh(Mesh):
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> NDArray:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return F.homo_entity_barycenter(entity, node)

    def bc_to_point(self, bcs: Union[NDArray, Sequence[NDArray]],
                    etype: Union[int, str]='cell', index: Index=_S) -> NDArray:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
        """
        node = self.entity('node')
        entity = self.entity(etype, index)
        order = getattr(entity, 'bc_order', None)
        return F.bc_to_points(bcs, node, entity, order)

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> NDArray:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        raise NotImplementedError

    def edge_to_ipoint(self, p: int, index: Index=_S) -> NDArray:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        indices = np.arange(NE, dtype=edges.dtype)[index]
        return np.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + np.arange(p-1, dtype=edges.dtype) + NN,
            edges[:, 1].reshape(-1, 1),
        ], axis=-1)


class SimplexMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return F.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        return F.simplex_gdof(p, self)

    # shape function
    def grad_lambda(self, index: Index=_S) -> NDArray:
        raise NotImplementedError

    def shape_function(self, bc: NDArray, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[NDArray]=None) -> NDArray:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.itype)
        phi = F.simplex_shape_function(bc, p, mi, dtype=self.ftype)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return  np.expand_dims(phi, axis=0)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

    def grad_shape_function(self, bc: NDArray, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[NDArray]=None) -> NDArray:
        TD = bc.shape[-1] - 1
        mi = mi or F.multi_index_matrix(p, TD, dtype=self.itype)
        R = F.simplex_grad_shape_function(bc, p, mi, dtype=self.ftype) # (NQ, ldof, bc)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            # gphi = np.einsum('...bm, kjb -> k...jm', Dlambda, R) # (NQ, NC, ldof, dim)
            gphi = np.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'k': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

