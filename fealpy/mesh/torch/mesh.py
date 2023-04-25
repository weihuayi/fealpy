
from abc import ABCMeta, abstractmethod
from typing import Union, Optional
import torch
from torch import Tensor, device
import numpy as np

from .mesh_data_structure import (
    MeshDataStructure,
    Mesh1dDataStructure,
    Mesh2dDataStructure,
    Mesh3dDataStructure
)


class DimensionError(Exception):
    pass


class Mesh(metaclass=ABCMeta):
    """
    @brief The abstract base class for all meshes in fealpy. This can not be\
           instantiated before all abstract methods are overriden.

    @note: Abstract methods list:
    ```
    def uniform_refine(self) -> None: ...
    def add_plot(self, axes) -> None: ...

    def integrator(self, k: int, etype: Union[int, str]): ...
    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> Tensor: ...
    @classmethod
    def multi_index_matrix(cls, p: int) -> Tensor: ...
    def shape_function(self, bc: Tensor, p: int) -> Tensor: ...
    def grad_shape_function(self, bc: Tensor, p: int, index=np.s_[:]) -> Tensor: ...
    ...
    ```
    """
    ds: MeshDataStructure
    node: Tensor

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
        return self.ds.cell.shape[-1]

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.ds.NEC

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.ds.NFC

    @abstractmethod
    def uniform_refine(self, n: int=1) -> None:
        """
        @brief Refine the whole mesh uniformly for `n` times.
        """
        pass

    # @abstractmethod
    # def show_function(self):
    #     pass

    # @abstractmethod
    # def show_animation(self):
    #     pass


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

    def bc_to_point(self, bc: Tensor, etype: Union[int, str]='cell',
                    index=np.s_[:]) -> Tensor:
        """
        @brief Convert barycenter coordinate points to cartesian coordinate points\
               on mesh entities.

        @param bc: Barycenter coordinate points tensor, with shape (NQ, NVC), where\
                   NVC is the number of nodes in each entity.
        @param etype: Specify the type of entities on which the coordinates be converted.
        @param index: Index to slice entities.

        @note: To get the correct result, the order of bc must match the order of nodes\
               in the entity.

        @return: Cartesian coordinate points tensor, with shape (NQ, GD).
        """
        if etype in {'node', 0}:
            raise ValueError(f"Can not convert the coordinates on nodes, please\
                             use type of entities of higher dimension.")
        node = self.node
        entity = self.entity(etype=etype, index=index)
        p = torch.einsum('...j, ijk -> ...ik', bc, node[entity])
        return p

    def entity(self, etype: Union[int, str], index=np.s_[:]) -> Tensor:
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

    def entity_barycenter(self, etype: Union[int, str], index=np.s_[:]) -> Tensor:
        """
        @brief Calculate barycenters of entities.
        """
        node = self.entity('node')
        TD = self.ds.TD
        if etype in {'cell', TD}:
            cell = self.ds.cell
            return torch.sum(node[cell[index], :], dim=1) / cell.shape[1]
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            return torch.sum(node[edge[index], :], dim=1) / edge.shape[1]
        elif etype in {'node', 0}:
            return node[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            face = self.ds.face
            return torch.sum(node[face[index], :], dim=1) / face.shape[1]
        raise ValueError(f"Invalid etype '{etype}'.")

    @abstractmethod
    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> Tensor:
        """
        @brief Calculate measurements of entities.
        """
        pass

    @staticmethod
    @abstractmethod
    def multi_index_matrix(p: int, etype: Union[int, str]='cell',
                           deivce: Optional[device]=None) -> Tensor:
        """
        @brief Make the multi-index matrix of a single entity.
        """
        pass

    @abstractmethod
    def shape_function(self, bc: Tensor, p: int) -> Tensor:
        """
        @brief
        """
        pass

    @abstractmethod
    def grad_shape_function(self, bc: Tensor, p: int, index=np.s_[:]) -> Tensor:
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
    def interpolation_points(self, p: int) -> Tensor:
        """
        @brief Get all the p-order interpolation points in the mesh.
        """
        pass

    @abstractmethod
    def cell_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def edge_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def face_to_ipoint(self, p: int, index=np.s_[:]):
        pass

    @abstractmethod
    def node_to_ipoint(self, p: int, index=np.s_[:]):
        pass


##################################################
### Mesh with Different Topology Dimension
##################################################


class Mesh1d(Mesh):
    """
    @brief The abstract class for all meshes with topology dimension 1.\
           This is still abstract, and some methods need to be overiden.
    """
    ds: Mesh1dDataStructure


class Mesh2d(Mesh):
    """
    @brief The abstract class for all meshes with topology dimension 2.\
           This is still abstract, and some methods need to be overiden.
    """
    ds: Mesh2dDataStructure

    def entity_measure(self, etype: Union[int, str]=2, index=np.s_[:]):
        """
        @brief Get measurements for entities.
        """
        if etype in {'cell', 2}:
            return self.cell_area()[index] #TODO: self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return torch.zeros((1,))
        raise ValueError(f"Invalid etity type {etype}.")

    # def _cell_area(self, index=np.s_[:]) -> Tensor:
    #     """
    #     @brief Area of cells in a 2-d mesh.
    #     """
    #     NC = self.number_of_cells()
    #     node = self.node
    #     edge = self.ds.edge
    #     edge2cell = self.ds.edge2cell
    #     is_inner_edge = ~self.ds.boundary_edge_flag()

    #     v = (node[edge[:, 1], :] - node[edge[:, 0], :])
    #     val = torch.linalg.vecdot(v, node[edge[:, 0], :], dim=-1)
    #     # TODO: torch.add.at?
    #     a = torch.bincount(edge2cell[:, 0], weights=val, minlength=NC)
    #     a += torch.bincount(edge2cell[is_inner_edge, 1], weights=-val[is_inner_edge], minlength=NC)
    #     a /= 2
    #     return a

    def cell_area(self, index=np.s_[:]) -> Tensor:
        """
        @brief Area of cells in a 2-d mesh.
        """
        GD = self.geo_dimension()
        NVC = self.number_of_nodes_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        v1 = node[cell[index, 1:NVC-1], :] - node[cell[index, 0:1], :]
        v2 = node[cell[index, 2:NVC], :] - node[cell[index, 1:NVC-1], :]

        if GD == 2:
            sub_area = (v1[..., 0]*v2[..., 1] - v2[..., 0]*v1[..., 1]) * 0.5
            return torch.sum(sub_area, dim=-1)
        elif GD == 3:
            sub_area = torch.linalg.cross(v1, v2, dim=-1)
            return torch.norm(torch.sum(sub_area, dim=-2), dim=-1)
        else:
            raise DimensionError(f"Unexcepted geometry dimension '{GD}' occured.")

    def edge_length(self, index=np.s_[:]) -> Tensor:
        """
        @brief Length of edges in a 2-d mesh.
        """
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index, 1], :] - node[edge[index, 0], :]
        return torch.norm(v, dim=-1)


class Mesh3d(Mesh):
    """
    @brief The abstract class for all meshes with topology dimension 3.\
           This is still abstract, and some methods need to be overiden.
    """
    ds: Mesh3dDataStructure
