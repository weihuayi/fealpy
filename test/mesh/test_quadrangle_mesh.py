
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh
from fealpy.geometry.utils import *

from quadrangle_mesh_data import *


class TestQuadrangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = QuadrangleMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"]
        assert mesh.number_of_edges() == meshdata["NE"]
        assert mesh.number_of_faces() == meshdata["NF"]
        assert mesh.number_of_cells() == meshdata["NC"]
        face2cell = mesh.face2cell
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

        mesh_from_one_quadrangle_1 = QuadrangleMesh.from_one_quadrangle()
        mesh_from_one_quadrangle_2 = QuadrangleMesh.from_one_quadrangle(meshtype='rectangle')
        mesh_from_one_quadrangle_3 = QuadrangleMesh.from_one_quadrangle(meshtype='rhombus')

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", box_data)
    def test_from_box(self, meshdata, backend):
        bm.set_backend(backend)
        box = meshdata['box']
        nx = meshdata['nx']
        ny = meshdata['ny']
        threshold = meshdata['threshold']
        mesh = QuadrangleMesh.from_box(box, nx, ny, threshold)

        node = mesh.node
        np.testing.assert_array_equal(bm.to_numpy(node), meshdata["node"])
        cell = mesh.cell
        np.testing.assert_array_equal(bm.to_numpy(cell), meshdata["cell"])
        edge = mesh.edge
        np.testing.assert_array_equal(bm.to_numpy(edge), meshdata["edge"])
        # # TODO: 添加 Threshold 时 jax 后端出错
        # face2cell = mesh.face2cell
        # np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", entity_data)
    def test_entity(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        q = meshdata['q']

        assert mesh.entity_measure(0) == meshdata["entity_measure"][0]
        assert all(mesh.entity_measure(1) == meshdata["entity_measure"][1])
        assert all(mesh.entity_measure('cell') == meshdata["entity_measure"][2])

        edge_barycenter = mesh.entity_barycenter('edge')
        cell_barycenter = mesh.entity_barycenter('cell')
        np.testing.assert_allclose(bm.to_numpy(edge_barycenter), meshdata["edge_barycenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(cell_barycenter), meshdata["cell_barycenter"], atol=1e-7)

        # TODO: have no boundary_edge_index
        boundary_node_index = mesh.boundary_node_index()
        boundary_cell_index = mesh.boundary_cell_index()
        boundary_face_index = mesh.boundary_face_index()
        # boundary_edge_index = mesh.boundary_edge_index()
        np.testing.assert_array_equal(bm.to_numpy(boundary_node_index), meshdata["boundary_node_index"])
        np.testing.assert_array_equal(bm.to_numpy(boundary_face_index), meshdata["boundary_node_index"])
        np.testing.assert_array_equal(bm.to_numpy(boundary_cell_index), meshdata["boundary_cell_index"])
        # np.testing.assert_array_equal(bm.to_numpy(boundary_edge_index), meshdata["boundary_edge_index"])

        integrator = mesh.quadrature_formula(q)
        bcs, ws = integrator.get_quadrature_points_and_weights()

        np.testing.assert_allclose(bm.to_numpy(bcs[0]), meshdata["bcs"][0], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(bcs[1]), meshdata["bcs"][1], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(ws), meshdata["ws"], atol=1e-7)

        point = mesh.bc_to_point(bcs)
        np.testing.assert_allclose(bm.to_numpy(point), meshdata["point"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", geo_data)
    def test_geo(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)

        edge_frame = mesh.edge_frame()
        edge_unit_normal = mesh.edge_unit_normal()

        np.testing.assert_allclose(bm.to_numpy(edge_frame[0]), meshdata["edge_frame"][0], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(edge_frame[1]), meshdata["edge_frame"][1], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(edge_unit_normal), meshdata["edge_unit_normal"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", cal_data)
    def test_cal_data(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        bcs = (bm.from_numpy(meshdata['bcs'][0]), bm.from_numpy(meshdata['bcs'][1]))
        mesh = QuadrangleMesh(node, cell)

        shape_function = mesh.shape_function(bcs)
        np.testing.assert_allclose(bm.to_numpy(shape_function), meshdata["shape_function"], atol=1e-7)
        grad_shape_function = mesh.grad_shape_function(bcs)
        np.testing.assert_allclose(bm.to_numpy(grad_shape_function), meshdata["grad_shape_function"], atol=1e-7)
        grad_shape_function_x = mesh.grad_shape_function(bcs, variables='x')
        np.testing.assert_allclose(bm.to_numpy(grad_shape_function_x), meshdata["grad_shape_function_x"], atol=1e-7)

        jacobi_matrix = mesh.jacobi_matrix(bcs)
        np.testing.assert_allclose(bm.to_numpy(jacobi_matrix), meshdata["jacobi_matrix"], atol=1e-7)
        first_fundamental_form = mesh.first_fundamental_form(jacobi_matrix)
        np.testing.assert_allclose(bm.to_numpy(first_fundamental_form), meshdata["first_fundamental_form"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", extend_data)
    def test_extend_data(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        p = meshdata["p"]

        assert mesh.number_of_global_ipoints(p) == meshdata["number_of_global_ipoints"]
        assert mesh.number_of_local_ipoints(p) == meshdata["number_of_local_ipoints"]
        assert mesh.number_of_corner_nodes() == meshdata["number_of_corner_nodes"]

        cell_to_ipoint = mesh.cell_to_ipoint(p)
        np.testing.assert_allclose(bm.to_numpy(cell_to_ipoint), meshdata["cell_to_ipoint"], atol=1e-7)

        interpolation_points = mesh.interpolation_points(p)
        np.testing.assert_allclose(bm.to_numpy(interpolation_points), meshdata["interpolation_points"], atol=1e-7)

        jacobi_at_corner = mesh.jacobi_at_corner()
        np.testing.assert_allclose(bm.to_numpy(jacobi_at_corner), meshdata["jacobi_at_corner"], atol=1e-7)

        angle = mesh.angle()
        np.testing.assert_allclose(bm.to_numpy(angle), meshdata["angle"], atol=1e-7)

        cell_quality = mesh.cell_quality()
        np.testing.assert_allclose(bm.to_numpy(cell_quality), meshdata["cell_quality"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", refine_data)
    def test_refine(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])
        mesh = QuadrangleMesh(node, cell)
        n = meshdata["n"]

        mesh.uniform_refine(n)
        refine_node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(refine_node), meshdata["refine_node"], atol=1e-7)
        refine_cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(refine_cell), meshdata["refine_cell"], atol=1e-7)
        refine_edge = mesh.edge
        np.testing.assert_allclose(bm.to_numpy(refine_edge), meshdata["refine_edge"], atol=1e-7)
        # refine_face_to_cell = mesh.face2cell
        # # TODO:  jax 后端出错，但 node 与 cell 都一致
        # np.testing.assert_allclose(bm.to_numpy(refine_face_to_cell),meshdata["refine_face_to_cell"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_from_polygon_gmsh_data)
    def test_mesh_from_polygon_gmsh(self, meshdata, backend):
        bm.set_backend(backend)
        vertices = meshdata['vertices']
        h = meshdata['h']
        mesh = QuadrangleMesh.from_polygon_gmsh(vertices, h)

        node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"], atol=1e-7)
        # TODO:  jax 后端 edge，face2cell 出错，但 node 与 cell 都一致
        # edge = mesh.edge
        # np.testing.assert_allclose(bm.to_numpy(edge), meshdata["edge"], atol=1e-7)
        cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(cell), meshdata["cell"], atol=1e-7)
        # face2cell = mesh.face2cell
        # np.testing.assert_allclose(bm.to_numpy(face2cell), meshdata["face2cell"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_from_triangle_data)
    def test_mesh_from_triangle(self, meshdata, backend):
        bm.set_backend(backend)
        from fealpy.mesh.triangle_mesh import TriangleMesh
        tri_node = bm.from_numpy(meshdata['tri_node'])
        tri_cell = bm.from_numpy(meshdata['tri_cell'])
        tri_mesh = TriangleMesh(tri_node, tri_cell)

        mesh = QuadrangleMesh.from_triangle_mesh(tri_mesh)

        node = mesh.node
        np.testing.assert_allclose(bm.to_numpy(node), meshdata["node"], atol=1e-7)
        # TODO:  jax 后端 edge，face2cell 出错，但 node 与 cell 都一致
        # edge = mesh.edge
        # np.testing.assert_allclose(bm.to_numpy(edge), meshdata["edge"], atol=1e-7)
        cell = mesh.cell
        np.testing.assert_allclose(bm.to_numpy(cell), meshdata["cell"], atol=1e-7)
        # face2cell = mesh.face2cell
        # np.testing.assert_allclose(bm.to_numpy(face2cell), meshdata["face2cell"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("meshdata", mesh_from_sub_domain_data)
    def test_mesh_from_sub_domain(self, meshdata, backend):
        bm.set_backend(backend)
        domain_node = bm.from_numpy(meshdata['domain_node'])
        domain_edge = bm.from_numpy(meshdata['domain_edge'])
        domain_line = bm.from_numpy(meshdata['domain_line'])
        domain_half_edge = bm.from_numpy(meshdata['domain_half_edge'])
        mesh = QuadrangleMesh.sub_domain_mesh_generator(domain_half_edge, domain_node, domain_line)
        cell_domain_tag = mesh.celldata['cell_domain_tag']
        print(cell_domain_tag)

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("meshdata", mesh_from_sub_domain_data)
    def test_sub_domain_divider(self, meshdata, backend):
        bm.set_backend(backend)
        domain_node = bm.from_numpy(meshdata['domain_node'])
        domain_edge = bm.from_numpy(meshdata['domain_edge'])
        domain_line = bm.from_numpy(meshdata['domain_line'])
        node_to_out_edge = bm.from_numpy(meshdata['node_to_out_edge'])

        total_line = []
        for l in domain_line:
            total_line.append(l)
            total_line.append(l[::-1])
        ne = next_edge_searcher(0, 1, node_to_out_edge, total_line)



if __name__ == "__main__":
    pytest.main(["./test_quadrangle_mesh.py", "-k", "test_init"])