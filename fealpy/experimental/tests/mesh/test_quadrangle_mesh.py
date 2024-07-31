import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.experimental.tests.mesh.quadrangle_mesh_data import *

# 测试不同的后端
# backends = ['numpy', 'pytorch', 'jax']
backends = ['numpy', 'pytorch']

class TestQuadrangleMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = QuadrangleMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"]
        assert mesh.number_of_edges() == meshdata["NE"]
        assert mesh.number_of_faces() == meshdata["NF"]
        assert mesh.number_of_cells() == meshdata["NC"]

if __name__ == "__main__":
    pytest.main(["./test_quadrangle_mesh.py"])