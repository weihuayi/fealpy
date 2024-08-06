import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.tests.mesh.triangle_mesh_data import *


class TestTriangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)

        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])

        mesh = TriangleMesh(node, cell)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", from_one_triangle_data)
    def test_from_one_triangle(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_one_triangle(meshtype='equ')

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        cell =  mesh.entity('cell')
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", from_box_data)
    def test_from_box(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx=2, ny=2)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        cell =  mesh.entity('cell')
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
 
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", entity_measure_data)
    def test_entity_measure(self, data, backend):
        bm.set_backend(backend)

        node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

        mesh = TriangleMesh(node, cell)
        nm = mesh.entity_measure('node')
        em = mesh.entity_measure('edge')
        cm = mesh.entity_measure('cell') 

        np.testing.assert_allclose(bm.to_numpy(nm), data["node_measure"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(em), data["edge_measure"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(cm), data["cell_measure"], atol=1e-14)    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", grad_lambda_data)
    def test_grad_lambda(self, data, backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=2, ny=2)
        val = mesh.grad_lambda()

        np.testing.assert_allclose(bm.to_numpy(val), data["val"], atol=1e-14)    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", grad_shape_function_data)
    def test_grad_shape_function(self, data, backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx=2, ny=2)
        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = mesh.grad_shape_function(bcs, p=2)

        np.testing.assert_allclose(bm.to_numpy(gphi), data["gphi"], atol=1e-14)    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", interpolation_point_data)
    def test_interpolation_points(self, data, backend):
        bm.set_backend(backend)
       
        mesh = TriangleMesh.from_box(nx=2, ny=2)
        mesh.itype=bm.int64
        ip = mesh.interpolation_points(4)
        cip = mesh.cell_to_ipoint(4)
        fip = mesh.face_to_ipoint(4)

        np.testing.assert_allclose(bm.to_numpy(ip), data["ips"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(cip), data["cip"], atol=1e-14)    
        np.testing.assert_allclose(bm.to_numpy(fip), data["fip"], atol=1e-14)    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", uniform_refine_data)
    def test_unifrom_refine(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_one_triangle()
        mesh.uniform_refine(2)
        
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        face2cell = mesh.face_to_cell()
        cell2edge = mesh.cell_to_edge()

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(cell), data["cell"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(face2cell), data["face2cell"], atol=1e-14)
        np.testing.assert_allclose(bm.to_numpy(cell2edge), data["cell2edge"], atol=1e-14)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", jacobian_matrix_data)
    def test_jacobian_matrix(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx=2, ny=2)

        jacobian = mesh.jacobian_matrix()

        np.testing.assert_allclose(bm.to_numpy(jacobian), data["jacobian_matrix"], atol=1e-14)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", from_unit_sphere_surface_data)
    def test_from_unit_sphere_surface(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_unit_sphere_surface()

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        cell =  mesh.entity('cell')
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

if __name__ == "__main__":
    #a = TestTriangleMeshInterfaces()
    #a.test_interpolation_points(interpolation_point_data[0], 'pytorch')
    pytest.main(["./test_triangle_mesh.py"])

