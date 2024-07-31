
import ipdb
import numpy as np 
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax']

# 定义多个典型的 TriangleMesh 对象
mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        "edge": None, 
        "cell": np.array([[0, 1, 2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 2, 2], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.int32),
        "NN": 3,
        "NE": 3,
        "NF": 3,
        "NC": 1
    },
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "edge": None,
        "cell": np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int32),
        "face2cell": None,
        "NN": 4,
        "NE": 5, 
        "NF": 5,
        "NC": 2
    }
]

class TestTriangleMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("mesh_data", mesh_data)
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


if __name__ == "__main__":
    pytest.main()
