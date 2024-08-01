
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
        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype=data['meshtype'])

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_faces() == data["NF"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), data["face2cell"])

if __name__ == "__main__":
    pytest.main(["./test_tetrahedron_mesh.py", "-k", "test_from_one_tetrahedron"])
