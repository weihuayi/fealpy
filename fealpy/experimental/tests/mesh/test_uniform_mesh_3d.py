import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.uniform_mesh_3d import UniformMesh3d
from fealpy.experimental.tests.mesh.uniform_mesh_3d_data import *

class TestUniformMesh3dInterfaces:

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)

        node = bm.to_numpy(mesh.node)
        node_true = meshdata['node']

        assert node.shape == node_true.shape, \
        "Node shapes do not match."
        np.testing.assert_allclose(node, node_true, atol=1e-8)

        edge = bm.to_numpy(mesh.edge)
        edge_true = meshdata['edge']
        
        assert edge.shape == edge_true.shape, \
            "Edge shapes do not match."
        np.testing.assert_allclose(edge, edge_true, atol=1e-8)

        face = bm.to_numpy(mesh.face)
        face_true = meshdata['face']

        assert face.shape == face_true.shape, \
            "Face shapes do not match."
        np.testing.assert_allclose(face, face_true, atol=1e-8)

        cell = bm.to_numpy(mesh.cell)
        cell_true = meshdata['cell']

        assert cell.shape == cell_true.shape, \
            "Cell shapes do not match."
        np.testing.assert_allclose(cell, cell_true, atol=1e-8)

        assert mesh.number_of_nodes() == meshdata['NN'], \
        "Number of nodes do not match."
        assert mesh.number_of_edges() == meshdata['NE'], \
        "Number of edges do not match."
        assert mesh.number_of_faces() == meshdata['NF'], \
        "Number of faces do not match."
        assert mesh.number_of_cells() == meshdata['NC'], \
        "Number of cells do not match."

    @pytest.mark.parametrize("meshdata", entity_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_entity(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)

        node = bm.to_numpy(mesh.entity('node'))
        node_true = meshdata['entity_node']
        np.testing.assert_almost_equal(node, node_true, decimal=7)

    @pytest.mark.parametrize("meshdata", entity_measure_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_entity_measure(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)

        assert len(mesh.entity_measure('edge')) == len(meshdata['edge_length']), \
        "Edge lengths must have the same length."
        for a, b in zip(mesh.entity_measure('edge'), meshdata['edge_length']):
            assert abs(a - b) < 1e-7, \
            f"Difference between {a} and {b} is greater than 1e-7"

        assert len(mesh.entity_measure('face')) == len(meshdata['face_area']), \
        "Face areas must have the same area."
        for a, b in zip(mesh.entity_measure('face'), meshdata['face_area']):
            assert abs(a - b) < 1e-7, \
            f"Difference between {a} and {b} is greater than 1e-7"

        assert (mesh.entity_measure('cell') - meshdata['cell_volume']) < 1e-7, \
        "Cell volumes are not as expected."

    @pytest.mark.parametrize("meshdata", uniform_refine_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_uniform_refine(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)

        mesh.uniform_refine(n=1)

        node_refined = mesh.node
        node_refined_true = meshdata['node_refined']

        np.testing.assert_allclose(node_refined, node_refined_true, atol=1e-8)