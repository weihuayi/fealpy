import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.tests.mesh.triangle_mesh_data import *

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax']

class TestTriangleMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):

        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = TriangleMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])
    
    @pytest.mark.parametrize("boxmeshdata", from_box_data)
    def test_from_box(self, boxmeshdata, backend):
        mesh = TriangleMesh.from_box(nx=2, ny=2)

        assert mesh.number_of_nodes() == boxmeshdata["NN"] 
        assert mesh.number_of_edges() == boxmeshdata["NE"] 
        assert mesh.number_of_faces() == boxmeshdata["NF"] 
        assert mesh.number_of_cells() == boxmeshdata["NC"] 
        
        cell =  mesh.entity('cell')
        np.testing.assert_array_equal(bm.to_numpy(cell), boxmeshdata["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), boxmeshdata["face2cell"])
    
    @pytest.mark.parametrize("mdata", entity_measure_data)
    def test_entity_measure(self, mdata, backend):
        node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

        mesh = TriangleMesh(node, cell)
        nm =  mesh.entity_measure('node')
        em = mesh.entity_measure('edge')
        cm = mesh.entity_measure('cell') 

        np.testing.assert_allclose(bm.to_numpy(nm), mdata["node_measure"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(em), mdata["edge_measure"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(cm), mdata["cell_measure"], atol=1e-14)    
    
    @pytest.mark.parametrize("gldata", grad_lambda_data)
    def test_grad_lambda(self, gldata, backend):
        mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=16, ny=16)
        flag = mesh.boundary_cell_flag()
        NC = bm.sum(flag)
        val = mesh.grad_lambda(index=flag)

        assert val.shape == gldata["val_shape"]


    @pytest.mark.parametrize("ipdata", interpolation_point_data)
    def test_interpolation_points(self, ipdata, backend):
        mesh = TriangleMesh.from_one_triangle()
        ip = mesh.interpolation_points(4)
        c2p = mesh.cell_to_ipoint(4)

        np.testing.assert_allclose(bm.to_numpy(ip), ipdata["ip"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(ip), ipdata["c2p"], atol=1e-14)

'''
    @pytest.mark.parametrize("meshdata", from_torus_surface_data)
    def test_from_torus_surface(self, meshdata, backend):
        R = meshdata["R"]
        r = meshdata["r"]
        Nu = meshdata["Nu"]
        Nv = meshdata["Nv"]
        expected_node_shape = meshdata["node_shape"]
        expected_cell_shape = meshdata["cell_shape"]

        mesh = TriangleMesh.from_torus_surface(R, r, Nu, Nv)

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        assert node.shape == expected_node_shape
        assert cell.shape == expected_cell_shape
''' 

if __name__ == "__main__":
    #a = TestTriangleMeshInterfaces()
    #a.test_from_box(from_box_data[0], 'numpy')
    pytest.main(["./test_triangle_mesh.py"])

