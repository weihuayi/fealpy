
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
    
if __name__ == "__main__":
    # pytest.main(["./test_prism_mesh.py", "-k", "test_init"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_one_prism"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_box"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_from_wedge"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_face_to_edge"])
    # pytest.main(["./test_prism_mesh.py", "-k", "test_number_of_faces"])
    pytest.main(["./test_prism_mesh.py", "-k", "test_number_of_ipoints"])