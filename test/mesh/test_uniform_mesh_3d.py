import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.uniform_mesh_3d import UniformMesh3d

from uniform_mesh_3d_data import *


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
        node_entity = bm.to_numpy(mesh.entity('node'))
        node_true = meshdata['node']
        np.testing.assert_almost_equal(node, node_true, decimal=7)
        np.testing.assert_almost_equal(node_entity, node_true, decimal=7)

        edge = bm.to_numpy(mesh.edge)
        edge_entity = bm.to_numpy(mesh.entity('edge'))
        edge_true = meshdata['edge']
        np.testing.assert_almost_equal(edge, edge_true, decimal=7)
        np.testing.assert_almost_equal(edge_entity, edge_true, decimal=7)


        face = bm.to_numpy(mesh.face)
        face_entity = bm.to_numpy(mesh.entity('face'))
        face_true = meshdata['face']
        np.testing.assert_almost_equal(face, face_true, decimal=7)
        np.testing.assert_almost_equal(face_entity, face_true, decimal=7)

        cell = bm.to_numpy(mesh.cell)
        cell_entity = bm.to_numpy(mesh.entity('cell'))
        cell_true = meshdata['cell']
        np.testing.assert_almost_equal(cell, cell_true, decimal=7)
        np.testing.assert_almost_equal(cell_entity, cell_true, decimal=7)

        assert mesh.number_of_nodes() == meshdata['NN'], \
        "Number of nodes do not match."
        assert mesh.number_of_edges() == meshdata['NE'], \
        "Number of edges do not match."
        assert mesh.number_of_faces() == meshdata['NF'], \
        "Number of faces do not match."
        assert mesh.number_of_cells() == meshdata['NC'], \
        "Number of cells do not match."


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

    
    @pytest.mark.parametrize("meshdata", barycenter_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_entity_barycenter(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)

        cell_barycenter = bm.to_numpy(mesh.cell_barycenter())
        cell_barycenter_true = meshdata['cell_barycenter']

        np.testing.assert_allclose(cell_barycenter, cell_barycenter_true, atol=1e-8)

    
    @pytest.mark.parametrize("meshdata", bc2point_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_bc_to_point(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh3d(extent, h, origin)
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        NF = mesh.number_of_faces()
        NE = mesh.number_of_edges()

        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        

        bc_to_point_cell = bm.to_numpy(mesh.bc_to_point(bcs=bcs))
        CNQ = ws.shape[0]
        ture_shape = (CNQ, NC, GD)
        assert bc_to_point_cell.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {bc_to_point_cell.shape}"

        bc_to_point_face = bm.to_numpy(mesh.bc_to_point(bcs=bcs[0:2]))
        FNQ = bcs[0].shape[0] * bcs[1].shape[0]
        ture_shape = (FNQ, NF, GD)
        assert bc_to_point_face.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {bc_to_point_face.shape}"

        bc_to_point_edge = bm.to_numpy(mesh.bc_to_point(bcs=bcs[0]))
        ENQ = bcs[0].shape[0]
        ture_shape = (ENQ, NE, GD)
        assert bc_to_point_edge.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {bc_to_point_edge.shape}"



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