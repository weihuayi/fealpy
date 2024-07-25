
from typing import Union, Optional, Dict, overload, Callable, Any

from ..backend import backend_manager as fealpy
from ..typing import TensorLike, Index, EntityName, _S, _int_func
from .. import logger
from .utils import estr2dim, edim2entity, MeshMeta


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

    def __init__(self, TD: int) -> None:
        assert hasattr(self, '_entity_dim_method_name_map')
        self._entity_storage: Dict[int, TensorLike] = {}
        self._entity_factory: Dict[int, Callable] = {
            k: getattr(self, self._entity_dim_method_name_map[k])
            for k in self._entity_dim_method_name_map
        }
        self.TD = TD

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

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> Any: return self.cell.dtype
    @property
    def device(self) -> Any: return self.cell.device
    def storage(self) -> Dict[int, TensorLike]:
        return self._entity_storage

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        entity = self.entity(etype)

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

    def _nv_entity(self, etype: Union[int, str]) -> TensorLike:
        entity = self.entity(etype)
        if hasattr(entity, 'location'):
            loc = entity.location
            return loc[1:] - loc[:-1]
        else:
            return fealpy.tensor((entity.shape[-1],), dtype=self.itype)

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
    def cell_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('cell', index)

    def face_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('face', index)

    def edge_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('edge', index)

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
        face2edge = cell2edge[
            face2cell[:, [0]],
            localFace2edge[face2cell[:, 2]]
        ]
        return face2edge[index]

    def cell_to_face(self, index: Index=_S) -> TensorLike:
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        NFC = self.number_of_faces_of_cells()

        face2cell = self.face2cell
        dtype = self.itype

        cell2face = fealpy.zeros((NC, NFC), dtype=dtype)
        arange_tensor = fealpy.arange(0, NF, dtype=dtype)

        assert cell2face.dtype == arange_tensor.dtype, f"Data type mismatch: cell2face is {cell2face.dtype}, arange_tensor is {arange_tensor.dtype}"

        cell2face[face2cell[:, 0], face2cell[:, 2]] = arange_tensor
        cell2face[face2cell[:, 1], face2cell[:, 3]] = arange_tensor
        return cell2face[index]

    def face_to_cell(self, index: Index=_S) -> TensorLike:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        return face2cell

    ### boundary
    def boundary_node_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = fealpy.zeros((NN,), **kwargs)
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
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = fealpy.zeros((NC,), **kwargs)
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

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalFace = self.total_face()
        _, i0_np, j_np = fealpy.unique(
            fealpy.sort(totalFace, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = fealpy.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = fealpy.arange(0, NFC*NC, dtype=i0_np.dtype)

        self.cell2face = j_np.reshape(NC, NFC)

        face2cell_np = fealpy.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = face2cell_np

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = fealpy.unique(
                fealpy.sort(total_edge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = j.reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell
            self.cell2edge = self.cell2face

        logger.info(f"Mesh toplogy relation constructed, with {NC} cells, {NF} "
                    f"faces, {NN} nodes "
                    f"on device ?")
