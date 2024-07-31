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
    
    @pytest.mark.parametrize("meshdata", from_box_data)
    def test_from_box(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = TriangleMesh.from_box(nx=2, ny=2)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])
        
    @pytest.mark.parametrize("meshdata", entity_measure_data)
    def test_entity_measure(self, meshdata, backend):
        node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

        mesh = TriangleMesh(node, cell)
        assert mesh.entity_measure(0) == meshdata["measure0"]
        assert mesh.entity_measure(1) == meshdata["measure1"]
        assert mesh.entity_measure('cell') == meshdata["measure"]

if __name__ == "__main__":
    #a = TestTriangleMeshInterfaces()
    #a.test_from_box(from_box_data[0], 'numpy')
    pytest.main(["./test_triangle_mesh.py"])

