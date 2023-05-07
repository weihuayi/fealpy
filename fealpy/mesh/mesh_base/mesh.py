"""
Provide the abstract base class for mesh.
"""

from abc import ABCMeta, abstractmethod
from typing import Union
from numpy.typing import NDArray
import numpy as np

from ..mesh_data_structure import MeshDataStructure


class Mesh(metaclass=ABCMeta):
    """
    @brief The abstract base class for mesh. This can not be instantiated before\
           all abstract methods being overriden.

    @note: Abstract methods list:
    ```
    def uniform_refine(self, n: int=1) -> int: ...
    def integrator(self, k: int, etype: Union[int, str]): ...
    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> NDArray: ...
    @staticmethod
    def multi_index_matrix(p: int, etype: Union[int, str]='cell') -> NDArray: ...
    def shape_function(self, bc: NDArray, p: int) -> NDArray: ...
    def grad_shape_function(self, bc: NDArray, p: int, index=np.s_[:]) -> NDArray: ...
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int: ...
    def number_of_global_ipoints(self, p: int) -> int: ...
    def interpolation_points(self, p: int) -> NDArray: ...
    def cell_to_ipoint(self, p: int, index=np.s_[:]): ...
    def face_to_ipoint(self, p: int, index=np.s_[:]): ...
    def edge_to_ipoint(self, p: int, index=np.s_[:]): ...
    def node_to_ipoint(self, p: int, index=np.s_[:]): ...
    """
    ds: MeshDataStructure
    node: NDArray
    itype: np.dtype
    ftype: np.dtype

    ### General Interfaces ###

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    def number_of_nodes_of_cells(self) -> int:
        """Number of nodes in a cell"""
        return self.ds.NVC

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.ds.NEC

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.ds.NFC

    number_of_vertices_of_cells = number_of_nodes_of_cells

    @abstractmethod
    def uniform_refine(self, n: int=1) -> None:
        """
        @brief Refine the whole mesh uniformly for `n` times.
        """
        pass


    ### FEM Interfaces ###

    def geo_dimension(self) -> int:
        """
        @brief Get geometry dimension of the mesh.
        """
        return self.node.shape[-1]

    def top_dimension(self) -> int:
        """
        @brief Get topology dimension of the mesh.
        """
        return self.ds.TD

    @abstractmethod
    def integrator(self, k: int, etype: Union[int, str]):
        """
        @brief Get the integration formula on a mesh entity of different dimensions.
        """
        pass

    def bc_to_point(self, bc: NDArray, etype: Union[int, str]='cell',
                    index=np.s_[:]) -> NDArray:
        """
        @brief Convert barycenter coordinate points to cartesian coordinate points\
               on mesh entities.

        @param bc: Barycenter coordinate points array, with shape (NQ, NVC), where\
                   NVC is the number of nodes in each entity.
        @param etype: Specify the type of entities on which the coordinates be converted.
        @param index: Index to slice entities.

        @note: To get the correct result, the order of bc must match the order of nodes\
               in the entity.

        @return: Cartesian coordinate points array, with shape (NQ, GD).
        """
        if etype in {'node', 0}:
            raise ValueError(f"Can not convert the coordinates on nodes, please\
                             use type of entities of higher dimension.")
        node = self.node
        entity = self.entity(etype=etype, index=index)
        p = np.einsum('...j, ijk -> ...ik', bc, node[entity])
        return p

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Get entities.

        @param etype: Type of entities. Accept dimension or name.
        @param index: Index for entities.

        @return: A tensor representing the entities in this mesh.
        """
        TD = self.ds.TD
        if etype in {'cell', TD}:
            return self.ds.cell[index]
        elif etype in {'edge', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, self.geo_dimension())[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            return self.ds.face[index]
        raise ValueError(f"Invalid etype '{etype}'.")

    def entity_barycenter(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate barycenters of entities.
        """
        node = self.entity('node')
        TD = self.ds.TD
        if etype in {'cell', TD}:
            cell = self.ds.cell
            return np.sum(node[cell[index], :], axis=1) / cell.shape[1]
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            return np.sum(node[edge[index], :], axis=1) / edge.shape[1]
        elif etype in {'node', 0}:
            return node[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            face = self.ds.face
            return np.sum(node[face[index], :], axis=1) / face.shape[1]
        raise ValueError(f"Invalid etype '{etype}'.")

    @abstractmethod
    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate measurements of entities.
        """
        pass

    @staticmethod
    @abstractmethod
    def multi_index_matrix(p: int, etype: Union[int, str]='cell') -> NDArray:
        """
        @brief Make the multi-index matrix of a single entity.
        """
        pass

    @abstractmethod
    def shape_function(self, bc: NDArray, p: int) -> NDArray:
        """
        @brief
        """
        pass

    @abstractmethod
    def grad_shape_function(self, bc: NDArray, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief
        """
        pass

    @abstractmethod
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        """
        @brief Return the number of p-order interpolation points in a single entity.
        """
        pass

    @abstractmethod
    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief Return the number of all p-order interpolation points.
        """
        pass

    @abstractmethod
    def interpolation_points(self, p: int) -> NDArray:
        """
        @brief Get all the p-order interpolation points in the mesh.
        """
        pass

    @abstractmethod
    def cell_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def face_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def edge_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def node_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    ### Other Interfaces ###

    def error(self, u, v, q=None, power=2, celltype=False):
        """
        @brief Calculate the error between two functions.
        """
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(u):
            if not hasattr(u, 'coordtype'):
                u = u(ps)
            else:
                if u.coordtype == 'cartesian':
                    u = u(ps)
                elif u.coordtype == 'barycentric':
                    u = u(bcs)

        if callable(v):
            if not hasattr(v, 'coordtype'):
                v = v(ps)
            else:
                if v.coordtype == 'cartesian':
                    v = v(ps)
                elif v.coordtype == 'barycentric':
                    v = v(bcs)

        if u.shape[-1] == 1:
            u = u[..., 0]

        if v.shape[-1] == 1:
            v = v[..., 0]

        cm = self.entity_measure('cell')

        f = np.power(np.abs(u - v), power)
        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, cm)

        if celltype is False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )
