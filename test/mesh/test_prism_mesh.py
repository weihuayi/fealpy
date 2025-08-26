
import numpy as np

import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh.prism_mesh import PrismMesh
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from prism_mesh_data import *

class TestPrismMeshInterfaces:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])

        mesh = PrismMesh(node, cell)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", total_face_data)
    def test_total_face(self, data, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])
        mesh = PrismMesh(node, cell)
        totalFace = mesh.total_face()
        np.testing.assert_array_equal(bm.to_numpy(totalFace), data["totalFace"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", from_one_prism_data)
    def test_from_one_prism(self, data, backend):
         bm.set_backend(backend)
         mesh = PrismMesh.from_one_prism(meshtype=data['meshtype'])

         assert mesh.number_of_nodes() == data["NN"] 
         assert mesh.number_of_edges() == data["NE"] 
         assert mesh.number_of_faces() == data["NF"] 
         assert mesh.number_of_cells() == data["NC"] 
        
         face2cell = mesh.face_to_cell()
         np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
    
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", from_box)
    def test_from_box(self, data, backend):
        bm.set_backend(backend)
        mesh = PrismMesh.from_box(box=[0,1,0,1,0,1], nx=1,ny=1,nz=2)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(edge), data["edge"])
        np.testing.assert_array_equal(bm.to_numpy(face), data["face"])
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", from_wedge_data)
    def test_from_wedge(self, data, backend):
        bm.set_backend(backend)
        tmesh = TriangleMesh.from_one_triangle()
        mesh = PrismMesh.from_wedge(tmesh=tmesh, h=1.0, nh=2)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-14)
        np.testing.assert_array_equal(bm.to_numpy(edge), data["edge"])
        np.testing.assert_array_equal(bm.to_numpy(face), data["face"])
        np.testing.assert_array_equal(bm.to_numpy(cell), data["cell"])

        face2cell = mesh.face_to_cell()
        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
   
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", face2edge_data)
    def test_face_to_edge(self, data, backend):
        bm.set_backend(backend)
        mesh = PrismMesh.from_one_prism()
        face2edge = mesh.face_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(face2edge), data["face2edge"])
    
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", NF_data)
    def test_number_of_faces(self, data, backend):
        bm.set_backend(backend)
        mesh = PrismMesh.from_box(nx=2,ny=2,nz=2)
        assert mesh.number_of_tri_faces() == data["NF_t"] 
        assert mesh.number_of_quad_faces() == data["NF_q"] 

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", ipoints_data)
    def test_number_of_ipoints(self, data, backend):
        bm.set_backend(backend)
        mesh = PrismMesh.from_box(nx=2,ny=2,nz=2)
        assert mesh.number_of_local_ipoints(p=3) == data["ldof"] 
        assert mesh.number_of_global_ipoints(p=3) == data["gdof"] 
    
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", entity2ipoint_data)
    def test_entity_to_point(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0],
                          [1.0, 1.0, 1.0],
                          [0.0, 1.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 3, 4, 5, 7],
                         [2, 3, 1, 6, 7, 5]], dtype=bm.int32)
        mesh = PrismMesh(node, cell)
        tri2ipoint = mesh.tri_to_ipoint(p=2)
        quad2ipoint = mesh.quad_to_ipoint(p=2)
        cell2ipoint = mesh.cell_to_ipoint(p=2)
        np.testing.assert_array_equal(bm.to_numpy(tri2ipoint), data["tface2ipoint"])
        np.testing.assert_array_equal(bm.to_numpy(quad2ipoint), data["qface2ipoint"])
        np.testing.assert_array_equal(bm.to_numpy(cell2ipoint), data["cell2ipoint"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", interpolation_data)
    def test_interpolation_points(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 3, 4, 5, 7],
                        [2, 3, 1, 6, 7, 5]], dtype=bm.int32)
        mesh = PrismMesh(node, cell)
        ipoint0 = mesh.interpolation_points(p=2)
        ipoint1 = mesh.interpolation_points(p=3)
        np.testing.assert_array_equal(bm.to_numpy(ipoint0), data["ipoint0"])
        np.testing.assert_allclose(bm.to_numpy(ipoint1), data["ipoint1"], atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", jacobi_matrix_data)
    def test_jacobi_matrix(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 3, 4, 5, 7],
                        [2, 3, 1, 6, 7, 5]], dtype=bm.int32)
        mesh = PrismMesh(node, cell)
        bc0 = bm.array([[1/3, 1/3, 1/3], 
                        [1/7, 2/7, 4/7],
                        [1/2, 1/2, 0],
                        [0, 1/4, 3/4]], dtype=bm.float64)
        bc1 = bm.array([[1/2, 1/2], 
                        [1/3, 2/3]], dtype=bm.float64)
        bcs = (bc0, bc1)
        J, gphi = mesh.jacobi_matrix(bcs, return_grad=True)

        np.testing.assert_allclose(bm.to_numpy(J), data["J"], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(gphi), data["gphi"], atol=1e-7, rtol=1e-7)
        # np.testing.assert_array_equal(bm.to_numpy(J), data["J"])
        # np.testing.assert_array_equal(bm.to_numpy(gphi), data["gphi"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", shape_function_data)
    def test_shape_function(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 3, 4, 5, 7],
                        [2, 3, 1, 6, 7, 5]], dtype=bm.int32)
        mesh = PrismMesh(node, cell)
        bc0 = bm.array([[1/3, 1/3, 1/3], 
                        [1/7, 2/7, 4/7],
                        [1/2, 1/2, 0],
                        [0, 1/4, 3/4]], dtype=bm.float64)
        bc1 = bm.array([[1/2, 1/2], 
                        [1/3, 2/3]], dtype=bm.float64)
        bcs = (bc0, bc1)
        val = mesh.shape_function(bcs)
        np.testing.assert_allclose(bm.to_numpy(val), data["phi"], atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", grad_shape_function_data)
    def test_grad_shape_function(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]], dtype=bm.float64)
        cell = bm.array([[0, 1, 3, 4, 5, 7],
                        [2, 3, 1, 6, 7, 5]], dtype=bm.int32)
        mesh = PrismMesh(node, cell)
        bc0 = bm.array([[1/3, 1/3, 1/3], 
                        [1/7, 2/7, 4/7],
                        [1/2, 1/2, 0],
                        [0, 1/4, 3/4]], dtype=bm.float64)
        bc1 = bm.array([[1/2, 1/2], 
                        [1/3, 2/3]], dtype=bm.float64)
        bcs = (bc0, bc1)
        val0 = mesh.grad_shape_function(bcs)
        val1 = mesh.grad_shape_function(bcs, variables='x')
        np.testing.assert_allclose(bm.to_numpy(val0), data["gphi0"], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(val1), data["gphi1"], atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", uniform_refine_data)
    def test_uniform_refine(self, data, backend):
        bm.set_backend(backend)
        node = bm.array([[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [0.0, 1.0, 1.0],
                 [0.0, 0.0, 2.0],
                 [1.0, 0.0, 2.0],
                 [0.0, 1.0, 2.0]])
        cell = bm.array([[0, 1, 2, 3, 4, 5],
                        [5, 3, 4, 8, 6, 7]])
        mesh = PrismMesh(node, cell)
        mesh.uniform_refine()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        np.testing.assert_allclose(bm.to_numpy(node), data["node"], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(cell), data["cell"], atol=1e-7, rtol=1e-7)

if __name__ == "__main__":
    # pytest.main(["./test_prism_mesh.py", "-k", "test_init"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_one_prism"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_box"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_wedge"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_face_to_edge"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_number_of_faces"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_entity_to_point"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_interpolation_points"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_shape_function"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_jacobi_matrix"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_grad_shape_function"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_uniform_refine"])
    pytest.main(["./test_prism_mesh.py"])