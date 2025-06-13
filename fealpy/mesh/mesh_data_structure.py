
from typing import Union, Optional, Dict, overload, Callable, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, EntityName, _S, _int_func
from .. import logger
from ..sparse import COOTensor, CSRTensor
from ..tools.sparse_tool import arr_to_csr
from .utils import estr2dim, edim2entity, MeshMeta, flocc


##################################################
### Mesh Data Structure Base
##################################################
# NOTE: MeshDS provides a storage for mesh entities and all topological methods.

class MeshDS(metaclass=MeshMeta):
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: TensorLike
    face: TensorLike
    edge: TensorLike
    node: TensorLike
    face2cell: TensorLike
    cell2edge: TensorLike
    localEdge: TensorLike # only for homogeneous mesh
    localFace: TensorLike # only for homogeneous mesh
    localFace2Edge: TensorLike

    def __init__(self, *, TD: int, itype, ftype) -> None:
        assert hasattr(self, '_entity_dim_method_name_map')
        self._entity_storage: Dict[int, TensorLike] = {}
        self._entity_factory: Dict[int, Callable] = {
            k: getattr(self, self._entity_dim_method_name_map[k])
            for k in self._entity_dim_method_name_map
        }
        self.TD = TD
        self.itype = itype
        self.ftype = ftype

    @overload
    def __getattr__(self, name: EntityName) -> TensorLike: ...
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

    def clear(self) -> None:
        """Remove all entities from the storage."""
        self._entity_storage.clear()

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def device(self) -> Any: return bm.get_device(self.cell)
    def storage(self) -> Dict[int, TensorLike]:
        return self._entity_storage

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        entity = self.entity(etype)

        if entity is None:
            logger.info(f'count: entity {etype} is not found and 0 is returned.')
            return 0

        if isinstance(entity, tuple):
            return entity[1].shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def _nv_entity(self, etype: Union[int, str]) -> TensorLike:
        entity = self.entity(etype)
        if isinstance(entity, tuple):
            loc = entity[1]
            return loc[1:] - loc[:-1]
        else:
            return bm.tensor((entity.shape[-1],), dtype=self.itype)

    def number_of_vertices_of_cells(self): return self._nv_entity('cell')
    def number_of_vertices_of_faces(self): return self._nv_entity('face')
    def number_of_vertices_of_edges(self): return self._nv_entity('edge')
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
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
    def cell_to_node(
        self, 
        index: Optional[Index] = None, 
        format: str = 'array'
    ) -> Union[TensorLike, CSRTensor, COOTensor]:
        if format == 'csr':
            return arr_to_csr(self.entity('cell', index))
        elif format == 'coo':  
            return arr_to_csr(self.entity('cell', index)).tocoo()
        elif format == 'array':
            return self.entity('cell', index)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )      

    def face_to_node(
        self, 
        index: Optional[Index] = None, 
        format: str = 'array'
    ) -> Union[TensorLike, CSRTensor, COOTensor]:
        if format == 'csr':
            return arr_to_csr(self.entity('face', index))
        elif format == 'coo':  
            return arr_to_csr(self.entity('face', index)).tocoo()
        elif format == 'array':
            return self.entity('face', index)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            ) 

    def edge_to_node(
        self, 
        index: Optional[Index] = None, 
        format: str = 'array'
    ) -> Union[TensorLike, CSRTensor, COOTensor]:
        if format == 'csr':
            return arr_to_csr(self.entity('edge', index))
        elif format == 'coo':  
            return arr_to_csr(self.entity('edge', index)).tocoo()
        elif format == 'array':
            return self.entity('edge', index)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )

    def cell_to_edge(self, index: Index=_S) -> TensorLike:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        return self.cell2edge[index]

    def face_to_edge(self, index: Index=_S):
        assert self.TD == 3
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[face2cell[:, [0]], localFace2edge[face2cell[:, 2]]]

        return face2edge[index]

    def cell_to_face(self, index: Index=_S) -> TensorLike:
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        NFC = self.number_of_faces_of_cells()

        face2cell = self.face2cell
        dtype = self.itype

        cell2face = bm.zeros((NC, NFC),device=bm.get_device(face2cell), dtype=dtype)
        arange_tensor = bm.arange(0, NF,device=bm.get_device(face2cell),dtype=dtype)

        assert cell2face.dtype == arange_tensor.dtype, f"Data type mismatch: cell2face is {cell2face.dtype}, arange_tensor is {arange_tensor.dtype}"

        cell2face = bm.set_at(cell2face, (face2cell[:, 0], face2cell[:, 2]), arange_tensor)
        cell2face = bm.set_at(cell2face, (face2cell[:, 1], face2cell[:, 3]), arange_tensor)
        return cell2face[index]

    def edge_to_cell(self, index: Index=_S) -> TensorLike:
        return self.face_to_cell(index)

    def face_to_cell(self, index: Index=_S) -> TensorLike:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        return face2cell

    def cell_to_cell(self):
        NC = self.number_of_cells()
        face2cell = self.face2cell
        NFC = self.number_of_faces_of_cells()
        cell2cell = bm.zeros((NC, NFC), dtype=self.itype)
        cell2cell = bm.set_at(cell2cell, (face2cell[:, 0], face2cell[:, 2]), face2cell[:, 1])
        cell2cell = bm.set_at(cell2cell, (face2cell[:, 1], face2cell[:, 3]), face2cell[:, 0])
        return cell2cell
    
    def node_to_node(self, format: str ='csr'):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edge = self.edge
        indice = bm.stack([edge.reshape(-1), edge[:, [1, 0]].reshape(-1)], axis=0)
        data = bm.ones((2*NE,),  dtype=bm.bool, device=self.device)
        if format == 'csr':
            node2node = COOTensor(indice, data, spshape=(NN, NN)).tocsr()
        elif format == 'coo':
            node2node = COOTensor(indice, data, spshape=(NN, NN))
        elif format == 'array':
            node2node = bm.zeros((NN, NN), dtype=bm.bool, device=self.device)
            node2node = bm.set_at(node2node, (indice[0], indice[1]), data)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )
        return node2node

    def node_to_edge(self, format: str ='csr'):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        nv = self.number_of_vertices_of_edges()
        edge = self.edge
        kwargs = bm.context(edge)
        indice = bm.stack([edge.reshape(-1), bm.repeat(bm.arange(NE,**kwargs), nv)], axis=0)
        data = bm.ones((NE*nv), dtype=bm.bool, device=self.device)
        if format == 'csr':
            node2edge = COOTensor(indice, data, spshape=(NN, NE)).tocsr()
        elif format == 'coo':
            node2edge = COOTensor(indice, data, spshape=(NN, NE))
        elif format == 'array':
            node2edge = bm.zeros((NN, NE), dtype=bm.bool, device=self.device)
            node2edge = bm.set_at(node2edge, (indice[0], indice[1]), data)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )  
        return node2edge

    def node_to_cell(self, format: str ='csr'):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        nv = self.number_of_vertices_of_cells()
        cell = self.cell
        kwargs = bm.context(cell)
        indice = bm.stack([cell.reshape(-1), bm.repeat(bm.arange(NC,**kwargs), nv)], axis=0)
        data = bm.ones((NC*nv),  dtype=bm.bool, device=self.device)
        if format == 'csr':
            node2cell = COOTensor(indice, data, spshape=(NN, NC)).tocsr()
        elif format == 'coo':
            node2cell = COOTensor(indice, data, spshape=(NN, NC))
        elif format == 'array':
            node2cell = bm.zeros((NN, NC), dtype=bm.bool, device=self.device)
            node2cell = bm.set_at(node2cell, (indice[0], indice[1]), data)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )  
        return node2cell
    
    def node_to_face(self, format: str ='csr'):
        NN = self.number_of_nodes()
        NF = self.number_of_faces()
        nv = self.number_of_vertices_of_faces()
        face = self.face
        kwargs = bm.context(face)
        indice = bm.stack([face.reshape(-1), bm.repeat(bm.arange(NF,**kwargs), nv)], axis=0)
        data = bm.ones((NF*nv),  dtype=bm.bool, device=self.device)
        if format == 'csr':
            node2face = COOTensor(indice, data, spshape=(NN, NF)).tocsr()
        elif format == 'coo':
            node2face = COOTensor(indice, data, spshape=(NN, NF))
        elif format == 'array':
            node2face = bm.zeros((NN, NF), dtype=bm.bool, device=self.device)
            node2face = bm.set_at(node2face, (indice[0], indice[1]), data)
        else:
            raise ValueError(
                f"Unsupported format: {format}."
                f"Must be one of 'array', 'csr', or 'coo'."
            )  
        return node2face

    ### boundary
    def boundary_node_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = bm.context(bd_face_flag)
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = bm.zeros((NN,), **kwargs)
        if bm.backend_name == "jax":
            bd_node_flag = bd_node_flag.at[bd_face2node.ravel()].set(True)
        else:
                bd_node_flag[bd_face2node.ravel()] = True
        return bd_node_flag

    def boundary_face_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            Tensor: boundary cell flag.
        """
        NC = self.number_of_cells()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = bm.zeros((NC,), **kwargs)
        if bm.backend_name == "jax":
            bd_cell_flag = bd_cell_flag.at[bd_face2cell.ravel()].set(True)
        else:
            bd_cell_flag[bd_face2cell.ravel()] = True
        return bd_cell_flag

    def boundary_node_index(self):
        return bm.nonzero(self.boundary_node_flag())[0]
    # TODO: finish this:
    # def boundary_edge_index(self):
    def boundary_face_index(self):
        return bm.nonzero(self.boundary_face_flag())[0]
    def boundary_cell_index(self):
        return bm.nonzero(self.boundary_cell_flag())[0]

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

    def total_face(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        NVE = local_edge.shape[-1]
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        totalFace = self.total_face()
        i0, i1, j = flocc(bm.sort(totalFace, axis=1))

        if self.TD > 1: # Do not add faces for interval mesh
            self.face = totalFace[i0, :] # this also adds the edge in 2-d meshes

        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()
        self.cell2face = bm.astype(j.reshape(NC, NFC), self.itype)
        self.face2cell = bm.astype(
            bm.stack([i0//NFC, i1//NFC, i0%NFC, i1%NFC], axis=-1),
            self.itype
        )
        # NOTE: dtype must be specified here, as these tensors are the results of unique.

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            totalEdge = self.total_edge()
            i2, _, j = flocc(bm.sort(totalEdge, axis=1))
            self.edge = totalEdge[i2, :]
            self.cell2edge = bm.astype(j.reshape(NC, NEC), self.itype)

        elif self.TD == 2:
            self.edge2cell = self.face2cell
            self.cell2edge = self.cell2face

        NN = self.number_of_nodes()
        NF = i0.shape[0]
        logger.info(f"Mesh toplogy relation constructed, with {NC} cells, {NF} "
                    f"faces, {NN} nodes "
                    f"on device ?")
