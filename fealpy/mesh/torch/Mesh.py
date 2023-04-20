
from abc import ABCMeta, abstractmethod
from typing import Union
import torch
from torch import Tensor
import numpy as np

from .mesh_data_structure import (
    MeshDataStructure,
    Mesh1dDataStructure,
    Mesh2dDataStructure
)


class Mesh(metaclass=ABCMeta):
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

    @abstractmethod
    def uniform_refine(self) -> None:
        """
        @brief Refine the whole mesh uniformly.
        """
        pass

    @abstractmethod
    def cell_location(self):
        pass

    @abstractmethod
    def show_function(self):
        pass

    @abstractmethod
    def show_animation(self):
        pass

    @abstractmethod
    def add_plot(self, axes):
        """
        @brief Plot the mesh in an axes.
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
    def integrator(self, k: int):
        pass

    @abstractmethod
    def bc_to_point(self, bc: Tensor) -> Tensor:
        pass

    def entity(self, etype: Union[int, str], index=np.s_[:]) -> Tensor:
        """
        @brief Get entities.
        """
        TD = self.ds.TD
        if etype in {'cell', TD}:
            return self.ds.cell[index]
        elif etype in {'face', TD-1}:
            return self.ds.face[index]
        elif etype in {'edge', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, self.geo_dimension())[index]
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

    @classmethod
    @abstractmethod
    def multi_index_matrix(cls, p: int) -> Tensor:
        """
        @brief Make the multi-index matrix of mesh.
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
        @brief Return the number of p-order integral points in a single entity.
        """
        pass

    @abstractmethod
    def number_of_global_ipoints(self, p: int):
        """
        @brief Return the number of all p-order integral points.
        """
        pass

    @abstractmethod
    def interpolation_points(self, p: int):
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


class _Entity():
    pass

# Similarly to the example above,
# Implement `_Measure`, `_Barycenter` class for every types of meshes.

class _Measure():
    pass

class _Barycenter():
    pass

# Mesh typs has `measure` and `barycenter` properties (point to the classes' instances above)
# to specify how to measure entities, and how to calculate the barycenter of entities.


class _Boundary():
    def __init__(self) -> None:
        pass

    def edge_flag(self):
        pass


class _Plot():
    pass


# Structure:
# mesh.torch Module
#  |- Mesh, Mesh1d, Mesh2d, Mesh3d -> mesh.py
#  |- MeshDataStructure, ... -> mesh_data_structure.py
#  |- Entity, Measure, Barcenter, ... -> typing.py
#  |- TriangleMesh, TriangleMeshDataStructure, _Entity, ... -> triangle_mesh.py


class Mesh1d(Mesh):
    ds: Mesh1dDataStructure


class Mesh2d(Mesh):
    ds: Mesh2dDataStructure

    def entity_measure(self, etype: Union[int, str]=2, index=np.s_[:]):
        """
        @brief Get measurements for entities.
        """
        if etype in {'cell', 2}:
            return self.cell_area()[index]
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        raise ValueError(f"Invalid etity type {etype}.")

    def cell_area(self) -> Tensor:
        """
        @brief Area of cells in a 2-d mesh.
        """
        NC = self.number_of_cells()
        node = self.node
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        is_inner_edge = ~self.ds.boundary_edge_flag()

        v = (node[edge[:, 1], :] - node[edge[:, 0], :])
        val = torch.sum(v*node[edge[:, 0], :], dim=1)
        a = torch.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a += torch.bincount(edge2cell[is_inner_edge, 1], weights=-val[is_inner_edge], minlength=NC)
        a /= 2
        return a

    def edge_length(self, index=np.s_[:]) -> Tensor:
        """
        @brief Length of edges in a 2-d mesh.
        """
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index, 1], :] - node[edge[index, 0], :]
        length = torch.sqrt(torch.sum(v**2, dim=1))
        return length
