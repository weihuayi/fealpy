
import ipdb
import numpy as np
import matplotlib.pyplot as plt

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.tetrahedron_mesh import TetrahedronMesh 
from fealpy.experimental.tests.mesh.tetrahedron_mesh_data import *


class TestTetrahedronMeshInterfaces:

    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])

        mesh = TetrahedronMesh(node, cell)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])
        
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", from_one_tetrahedron_data)
    def test_from_one_tetrahedron(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype=data['meshtype'])

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_to_edge_sign_data)
    def test_face_to_edge_sign(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
        sign = mesh.face_to_edge_sign() 
        np.testing.assert_array_equal(bm.to_numpy(sign), data["sign"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", face_unit_norm)
    def test_face_init_norm(self, data, backend):
        bm.set_backend(backend)
        np.set_printoptions(precision=16)
        mesh = TetrahedronMesh.from_box(box=[0,1,0,1,0,1], nx=3,ny=2,nz=1)
        n = mesh.face_unit_normal()
        np.testing.assert_allclose(bm.to_numpy(n), data["fn"], atol=1e-14)

if __name__ == "__main__":
    pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_init"])
    pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_from_one_tetrahedron"])
    pytest.main(["./test_tetrahedron_mesh.py", "-k",
        "test_from_face_to_edge_sign"])
