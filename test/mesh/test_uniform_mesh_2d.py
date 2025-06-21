import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d

from uniform_mesh_2d_data import *


class TestUniformMesh2dInterfaces:

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

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
        mesh = UniformMesh2d(extent, h, origin)

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
        mesh = UniformMesh2d(extent, h, origin)

        cell_barycenter = bm.to_numpy(mesh.cell_barycenter())
        edgex_barycenter = bm.to_numpy(mesh.edgex_barycenter())
        edgey_barycenter = bm.to_numpy(mesh.edgey_barycenter())

        cell_barycenter_true = meshdata['cell_barycenter']
        edgex_barycenter_true = meshdata['edgex_barycenter']
        edgey_barycenter_true = meshdata['edgey_barycenter']

        np.testing.assert_allclose(cell_barycenter, cell_barycenter_true, atol=1e-8)
        np.testing.assert_allclose(edgex_barycenter, edgex_barycenter_true, atol=1e-8)
        np.testing.assert_allclose(edgey_barycenter, edgey_barycenter_true, atol=1e-8)

    
    @pytest.mark.parametrize("meshdata", bc2point_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_bc_to_point(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        

        bc_to_point_cell = bm.to_numpy(mesh.bc_to_point(bcs=bcs))
        CNQ = ws.shape[0]
        ture_shape = (NC, CNQ, GD)
        assert bc_to_point_cell.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {bc_to_point_cell.shape}"

        bc_to_point_edge = bm.to_numpy(mesh.bc_to_point(bcs=bcs[0]))
        ENQ = bcs[0].shape[0]
        ture_shape = (NE, ENQ, GD)
        assert bc_to_point_edge.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {bc_to_point_edge.shape}"


    @pytest.mark.parametrize("meshdata", interpolation_points_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_interpolation_points(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        ipoints_p1 = mesh.interpolation_points(p=1)
        ipoints_p2 = mesh.interpolation_points(p=2)

        ipoints_p1_true = meshdata['ipoints_p1']
        ipoints_p2_true = meshdata['ipoints_p2']

        np.testing.assert_almost_equal(ipoints_p1, ipoints_p1_true, decimal=7)
        np.testing.assert_almost_equal(ipoints_p2, ipoints_p2_true, decimal=7)

    
    @pytest.mark.parametrize("meshdata", entity_to_ipoints_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_entity_to_ipoints(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        cell2ipoints_p1 = mesh.cell_to_ipoint(p=1)
        ture_shape = (mesh.number_of_cells(), mesh.number_of_local_ipoints(p=1))
        assert cell2ipoints_p1.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {cell2ipoints_p1.shape}"

        cell2ipoints_p2 = mesh.cell_to_ipoint(p=2)
        ture_shape = (mesh.number_of_cells(), mesh.number_of_local_ipoints(p=2))
        assert cell2ipoints_p2.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {cell2ipoints_p2.shape}"


    @pytest.mark.parametrize("meshdata", quadrature_formula_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_quadrature_formula(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

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

    @pytest.mark.parametrize("meshdata", shape_function_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_shape_function(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = ws.shape[0]

        p = 1
        shape_function_u_p1 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=p, variables='u'))
        ldof = mesh.number_of_local_ipoints(p=p)
        ture_shape = (NQ, ldof)
        assert shape_function_u_p1.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {shape_function_u_p1.shape}"

        shape_function_x_p1 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=p, variables='x'))
        ture_shape = (1, NQ, ldof)
        assert shape_function_x_p1.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {shape_function_x_p1.shape}"

        p = 2
        shape_function_u_p2 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=p, variables='u'))
        ldof = mesh.number_of_local_ipoints(p=p)
        ture_shape = (NQ, ldof)
        assert shape_function_u_p2.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {shape_function_u_p2.shape}"

        shape_function_x_p2 = bm.to_numpy(mesh.shape_function(bcs=bcs, p=p, variables='x'))
        ture_shape = (1, NQ, ldof)
        assert shape_function_x_p2.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {shape_function_x_p2.shape}"


    @pytest.mark.parametrize("meshdata", grad_shape_function_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_grad_shape_function(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()

        qf = mesh.quadrature_formula(q=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = ws.shape[0]

        p = 1
        grad_shape_function_u_p1 = bm.to_numpy(mesh.grad_shape_function(bcs=bcs, p=p, variables='u'))
        ldof = mesh.number_of_local_ipoints(p=p)
        ture_shape = (NQ, ldof, GD)
        assert grad_shape_function_u_p1.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {grad_shape_function_u_p1.shape}"

        grad_shape_function_x_p1 = bm.to_numpy(mesh.grad_shape_function(bcs=bcs, p=p, variables='x'))
        ture_shape = (NQ, NC, ldof, GD)
        assert grad_shape_function_x_p1.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {grad_shape_function_x_p1.shape}"

        p = 2
        gard_shape_function_u_p2 = bm.to_numpy(mesh.grad_shape_function(bcs=bcs, p=p, variables='u'))
        ldof = mesh.number_of_local_ipoints(p=p)
        ture_shape = (NQ, ldof, GD)
        assert gard_shape_function_u_p2.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {gard_shape_function_u_p2.shape}"

        grad_shape_function_x_p2 = bm.to_numpy(mesh.grad_shape_function(bcs=bcs, p=p, variables='x'))
        ture_shape = (NQ, NC, ldof, GD)
        assert grad_shape_function_x_p2.shape == ture_shape, \
        f"Expected shape {ture_shape}, but got {grad_shape_function_x_p2.shape}"


    @pytest.mark.parametrize("meshdata", topology_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_topology_data(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        edge2cell = bm.to_numpy(mesh.edge_to_cell())
        edge2cell_true = meshdata['edge2cell']

        # cell2cell = bm.to_numpy(mesh.cell_to_cell())
        # print("cell2cell:", cell2cell)
        # cell2cell_true = meshdata['cell2cell']
        # print("cell2cell_true:", cell2cell_true)

        np.testing.assert_allclose(edge2cell, edge2cell_true, atol=1e-8)
        # np.testing.assert_allclose(cell2cell, cell2cell_true, atol=1e-8)

    @pytest.mark.parametrize("meshdata", boundary_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_boundary_data(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

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
        mesh = UniformMesh2d(extent, h, origin)

        mesh.uniform_refine(n=1)

        node_refined = mesh.node
        node_refined_true = meshdata['node_refined']

        np.testing.assert_allclose(node_refined, node_refined_true, atol=1e-8)
