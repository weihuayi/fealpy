import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d

from uniform_mesh_1d_data import *


class TestUniformMesh1dInterfaces:

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

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
        mesh = UniformMesh1d(extent, h, origin)

        edge_length = bm.to_numpy(mesh.entity_measure('edge'))
        cell_area = bm.to_numpy(mesh.entity_measure('cell'))

        assert mesh.number_of_edges() == len(edge_length), \
        "Edge lengths must have the same length."
        assert mesh.number_of_cells() == len(cell_area), \
        "Cell areas must have the same length."

    
    @pytest.mark.parametrize("meshdata", barycenter_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_entity_barycenter(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

        cell_barycenter = bm.to_numpy(mesh.cell_barycenter())
        edge_barycenter = bm.to_numpy(mesh.edge_barycenter())

        cell_barycenter_true = meshdata['cell_barycenter']
        edge_barycenter_true = meshdata['edge_barycenter']

        np.testing.assert_allclose(cell_barycenter, cell_barycenter_true, atol=1e-8)
        np.testing.assert_allclose(edge_barycenter, edge_barycenter_true, atol=1e-8)

    @pytest.mark.parametrize("meshdata", interpolation_points_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_interpolation_points(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

        ipoints_p1 = mesh.interpolation_points(p=1)
        ipoints_p2 = mesh.interpolation_points(p=2)

        ipoints_p1_true = meshdata['ipoints_p1']
        ipoints_p2_true = meshdata['ipoints_p2']

        np.testing.assert_almost_equal(ipoints_p1, ipoints_p1_true, decimal=7)
        np.testing.assert_almost_equal(ipoints_p2, ipoints_p2_true, decimal=7)

    @pytest.mark.parametrize("meshdata", quadrature_formula_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_quadrature_formula(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

        qf1 = mesh.quadrature_formula(q=1)
        bcs_qf1, ws_qf1 = qf1.get_quadrature_points_and_weights()
        qf2 = mesh.quadrature_formula(q=2)
        bcs_qf2, ws_qf2 = qf2.get_quadrature_points_and_weights()

        assert len(bcs_qf1) == len(meshdata['bcs_q1']), "The tuples for qf1 must have the same length."
        for a, b in zip(bcs_qf1, meshdata['bcs_q1']):
            assert np.all(np.abs(bm.to_numpy(a) - b) < 1e-7), f"Difference in quadrature points for qf1 between {a} and {b} is greater than 1e-7"

        assert len(bcs_qf2) == len(meshdata['bcs_q2']), "The tuples for qf2 must have the same length."
        for a, b in zip(bcs_qf2, meshdata['bcs_q2']):
            assert np.all(np.abs(bm.to_numpy(a) - b) < 1e-7), f"Difference in quadrature points for qf2 between {a} and {b} is greater than 1e-7"

    @pytest.mark.parametrize("meshdata", boundary_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_boundary_data(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

        boundary_node_flag = bm.to_numpy(mesh.boundary_node_flag())
        boundary_node_flag_ture = meshdata['boundary_node_flag']
        boundary_edge_flag = bm.to_numpy(mesh.boundary_edge_flag())
        boundary_edge_flag_true = meshdata['boundary_edge_flag']
        boundary_cell_flag = bm.to_numpy(mesh.boundary_cell_flag())
        boundary_cell_flag_true = meshdata['boundary_cell_flag']

        np.testing.assert_allclose(boundary_node_flag, boundary_node_flag_ture, 
                                atol=1e-8)
        np.testing.assert_allclose(boundary_edge_flag, boundary_edge_flag_true, 
                                atol=1e-8)
        np.testing.assert_allclose(boundary_cell_flag, boundary_cell_flag_true, 
                                atol=1e-8)

    @pytest.mark.parametrize("meshdata", uniform_refine_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_uniform_refine(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh1d(extent, h, origin)

        mesh.uniform_refine(n=1)

        node_refined = mesh.node
        node_refined_true = meshdata['node_refined']

        np.testing.assert_allclose(node_refined, node_refined_true, atol=1e-8)

