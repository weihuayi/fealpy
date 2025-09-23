from typing import Type

import pytest
import numpy as np
import importlib
import matplotlib.pyplot as plt

from fealpy.cgraph.mesh import CreateMesh, Box3d, CircleMesh


def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)

def plot_2d(mesh):
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    # mesh.add_plot(axes, showaxis=True)
    # mesh.find_node(axes, showindex=True)
    # mesh.find_edge(axes, showindex=True)
    # mesh.find_cell(axes, showindex=True)
    plt.show()
        
def plot_3d(mesh):
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    mesh.add_plot(axes)
    # mesh.find_node(axes, showindex=True)
    # mesh.find_cell(axes, showindex=True)
    plt.show()

class TestMesh:
    @pytest.mark.parametrize("mesh_type", ["triangle", "quadrangle", 
                                "tetrahedron", "hexahedron", "edge"])
    def test_create_mesh(self, mesh_type):
        if mesh_type == "triangle":
            node = np.array([[0, 0], [1, 0], [0, 1]])
            cell = np.array([[0, 1, 2]])
        elif mesh_type == "quadrangle":
            node = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
            cell = np.array([[0, 2, 3, 1]])
            # node = np.array([[0,0], [1,0], [2,0],[2,1], 
            #         [2,2], [1,2], [0,2], [0,1], [1,1]]) # (NN,2)
            # cell = np.array([[1, 8, 7, 0], [2, 3, 8, 1], 
            #         [3, 4, 5, 8], [8, 5, 6 ,7]]) # (NC,4)
        elif mesh_type == "tetrahedron":
            node = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],
                    [0.5, np.sqrt(3)/2, 0.0],[0.5, np.sqrt(3)/6, np.sqrt(2/3)]])
            cell = np.array([[0, 1, 2, 3]])
        elif mesh_type == "hexahedron":
            node = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[1.0, 1.0, 0.0],[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 1.0, 1.0],[0.0, 1.0, 1.0]])
            cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        elif mesh_type == "edge":
            node = np.array([
                [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
                [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
                [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], [-2540, -2540, 0]])
            cell = np.array([
                [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
                [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
                [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
                [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
                [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]])
        else:
            raise ValueError(f"Test nodes not defined for mesh_type={mesh_type}")
        
        MeshClass = get_mesh_class(mesh_type)
        mesh = CreateMesh.run(mesh_type, node=node, cell=cell)

        assert isinstance(mesh, MeshClass)
        assert hasattr(mesh, "node")
        assert hasattr(mesh, "cell")
        assert mesh.node.shape[0] == node.shape[0]
        
        if mesh_type in [ "triangle","quadrangle"]:
            plot_2d(mesh)
        elif mesh_type in ["tetrahedron","hexahedron","edge"]:
            plot_3d(mesh)

    @pytest.mark.parametrize("mesh_type", ["tetrahedron", "hexahedron"])
    def test_box3d(self, mesh_type):
        domain = (0, 1, 0, 1, 0, 1)
        nx, ny, nz = 2, 2, 2

        MeshClass = get_mesh_class(mesh_type)
        mesh = Box3d.run(mesh_type, domain, nx, ny, nz)

        assert isinstance(mesh, MeshClass)
        assert hasattr(mesh, "node")
        assert hasattr(mesh, "cell")
        
        if mesh_type in ["tetrahedron", "hexahedron"]:
            plot_3d(mesh)
            
    @pytest.mark.parametrize("mesh_type", ["triangle"])
    def test_circle(self, mesh_type):
        X, Y = 1.2, 1.0
        radius = 2.0
        h = 0.5
        MeshClass = get_mesh_class(mesh_type)
        mesh = CircleMesh.run(mesh_type, X, Y, radius, h)
        
        assert isinstance(mesh, MeshClass)
        assert hasattr(mesh, "node")
        assert hasattr(mesh, "cell")
        
        if mesh_type == "triangle":
            plot_2d(mesh)
        

if __name__ == "__main__":
    a = TestMesh()
    a.test_create_mesh('triangle')
    a.test_create_mesh('quadrangle')
    a.test_create_mesh('tetrahedron')
    a.test_create_mesh('hexahedron')
    a.test_create_mesh('edge')
    a.test_box3d('tetrahedron')
    a.test_box3d('hexahedron')
    a.test_circle('triangle')
    # pytest.main(["test/cgraph/test_mesh.py"])