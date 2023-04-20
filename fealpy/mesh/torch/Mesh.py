
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

    @abstractmethod
    def add_plot(self, *args, **kwargs):
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

    def add_plot(
            self, plot_or_axes,
            nodecolor='w', edgecolor='k',
            cellcolor=[0.5, 0.9, 0.45], aspect=None,
            linewidths=1, markersize=50,
            showaxis=False, showcolorbar=False,
            cmax=None, cmin=None,
            colorbarshrink=1, cmap='jet', box=None
        ):
        pass
