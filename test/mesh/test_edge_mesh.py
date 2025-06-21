import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.edge_mesh import EdgeMesh

from edge_mesh_data import *


class TestEdgeMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = EdgeMesh(node, cell)
        
        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_cells() == meshdata["NC"] 
        
        cell2node = mesh.cell_to_node()
        
        
        np.testing.assert_array_equal(bm.to_numpy(cell2node), meshdata["cell2node"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_edge_tangent(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = EdgeMesh(node, cell)
        edge_tangent = mesh.edge_tangent()
        np.testing.assert_array_equal(bm.to_numpy(edge_tangent), meshdata["edge_tangent"])


    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_edge_length(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = EdgeMesh(node, cell)
        edge_length = mesh.edge_length()
        np.testing.assert_allclose(bm.to_numpy(edge_length), meshdata["edge_length"], atol=1e-14)
    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_grad_lambda(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = EdgeMesh(node,cell)
        grad_lambda = mesh.grad_lambda()
        np.testing.assert_array_equal(bm.to_numpy(grad_lambda), meshdata["grad_lambda"])

 
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_interpolation_points(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = EdgeMesh(node,cell)
        interpolation_points = mesh.interpolation_points(p=2)
        np.testing.assert_array_equal(bm.to_numpy(interpolation_points), meshdata["interpolation_points"])
        

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_cell_normal(self,meshdata,backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = EdgeMesh(node,cell)
        cell_normal = mesh.cell_normal()
        np.testing.assert_array_equal(bm.to_numpy(cell_normal), meshdata["cell_normal"])


if __name__ == "__main__":
   pytest.main(["test_edge_mesh.py"])





    






