
import ipdb
import numpy as np 
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax']

class TestTriangleMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    def test_init():
        node = bm.tensor([[0, 0], [1, 0], [0, 1]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)

        mesh = TriangleMesh(node, cell)

        assert mesh.node.shape == (3, 2)
        assert mesh.cell.shape == (1, 3)

        assert mesh.number_of_nodes() == 3
        assert mesh.number_of_cells() == 1

if __name__ == "__main__":
    pytest.main()
