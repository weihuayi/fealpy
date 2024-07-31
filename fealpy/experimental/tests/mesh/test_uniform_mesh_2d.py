import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.experimental.tests.mesh.uniform_mesh_2d_data import * 

# Test different backends
backends = ['numpy', 'pytorch', 'mindspore']  # TODO: Add 'jax' backend later

class TestUniformMesh2dInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_init(self, meshdata, backend):
        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        assert all((mesh.node.reshape(-1) - bm.from_numpy(meshdata['node']).reshape(-1)) < 1e-7)
        assert all(mesh.edge.reshape(-1) == bm.from_numpy(meshdata['edge']).reshape(-1))
        assert all(mesh.face.reshape(-1) == bm.from_numpy(meshdata['face']).reshape(-1))
        assert all(mesh.cell.reshape(-1) == bm.from_numpy(meshdata['cell']).reshape(-1))

        assert mesh.node.shape == meshdata['node'].shape
        assert mesh.edge.shape == meshdata['edge'].shape
        assert mesh.face.shape == meshdata['face'].shape
        assert mesh.cell.shape == meshdata['cell'].shape

        assert mesh.number_of_nodes() == meshdata['NN']
        assert mesh.number_of_edges() == meshdata['NE']
        assert mesh.number_of_faces() == meshdata['NF']
        assert mesh.number_of_cells() == meshdata['NC']

        assert len(mesh.entity_measure('edge')) == len(meshdata['edge_length']), "The tuples must have the same length."
        for a, b in zip(mesh.entity_measure('edge'), meshdata['edge_length']):
            assert abs(a - b) < 1e-7, f"Difference between {a} and {b} is greater than 1e-7"

        assert (mesh.entity_measure('cell') - meshdata['cell_area']) < 1e-7


if __name__ == "__main__":
    pytest.main()
