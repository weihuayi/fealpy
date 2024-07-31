import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.experimental.tests.mesh.uniform_mesh_2d_data import *

# Test different backends
backends = ['numpy', 'pytorch']  # TODO: Add 'jax' backend later

class TestUniformMesh2dInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.fixture
    def mesh(self, meshdata):
        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        return mesh

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_init(self, mesh, meshdata):
        assert all((mesh.node.reshape(-1) - bm.from_numpy(meshdata['node']).reshape(-1)) < 1e-7), "Mesh nodes are not as expected."
        assert all(mesh.edge.reshape(-1) == bm.from_numpy(meshdata['edge']).reshape(-1)), "Mesh edges are not as expected."
        assert all(mesh.face.reshape(-1) == bm.from_numpy(meshdata['face']).reshape(-1)), "Mesh faces are not as expected."
        assert all(mesh.cell.reshape(-1) == bm.from_numpy(meshdata['cell']).reshape(-1)), "Mesh cells are not as expected."

        assert mesh.node.shape == meshdata['node'].shape, "Node shapes do not match."
        assert mesh.edge.shape == meshdata['edge'].shape, "Edge shapes do not match."
        assert mesh.face.shape == meshdata['face'].shape, "Face shapes do not match."
        assert mesh.cell.shape == meshdata['cell'].shape, "Cell shapes do not match."

        assert mesh.number_of_nodes() == meshdata['NN'], "Number of nodes do not match."
        assert mesh.number_of_edges() == meshdata['NE'], "Number of edges do not match."
        assert mesh.number_of_faces() == meshdata['NF'], "Number of faces do not match."
        assert mesh.number_of_cells() == meshdata['NC'], "Number of cells do not match."

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_entity(self, mesh, meshdata):
        assert all((mesh.entity('node').reshape(-1) - bm.from_numpy(meshdata['entity_node']).reshape(-1)) < 1e-7), "Entity nodes do not match."

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_entity_measure(self, mesh, meshdata):
        assert len(mesh.entity_measure('edge')) == len(meshdata['edge_length']), "Edge lengths must have the same length."
        for a, b in zip(mesh.entity_measure('edge'), meshdata['edge_length']):
            assert abs(a - b) < 1e-7, f"Difference between {a} and {b} is greater than 1e-7"

        assert (mesh.entity_measure('cell') - meshdata['cell_area']) < 1e-7, "Cell areas do not match."

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_interpolation_points(self, mesh, meshdata):
        ipoints_p1 = mesh.interpolation_points(p=1)
        ipoints_p2 = mesh.interpolation_points(p=2)

        assert all((ipoints_p1.reshape(-1) - bm.from_numpy(meshdata['ipoints_p1']).reshape(-1)) < 1e-7), "Interpolation points p=1 do not match."
        assert all((ipoints_p2.reshape(-1) - bm.from_numpy(meshdata['ipoints_p2']).reshape(-1)) < 1e-7), "Interpolation points p=2 do not match."

    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_quadrature_formula(self, mesh, meshdata):
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
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_shape_function(self, mesh, meshdata):
        qf1 = mesh.quadrature_formula(q=1)
        bcs_qf1, ws_qf1 = qf1.get_quadrature_points_and_weights()

        shape_funciton_p1 = mesh.shape_function(bcs=bcs_qf1, p=1)
        shape_funciton_p2 = mesh.shape_function(bcs=bcs_qf1, p=2)
        assert all((shape_funciton_p1.reshape(-1) - bm.from_numpy(meshdata['shape_function_p1']).reshape(-1)) < 1e-7), "Shape function do not match."
        assert all((shape_funciton_p2.reshape(-1) - bm.from_numpy(meshdata['shape_function_p2']).reshape(-1)) < 1e-7), "Shape function do not match."
