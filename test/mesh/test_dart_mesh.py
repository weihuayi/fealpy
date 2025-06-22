import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh, DartMesh, TriangleMesh
from fealpy.geometry.utils import *

from dart_mesh_data import *


class TestDartMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init_and_from_mesh(self, meshdata, backend):
        bm.set_backend(backend)
        hex_node = bm.from_numpy(meshdata['hex_node'])
        hex_cell = bm.from_numpy(meshdata['hex_cell'])
        hex_mesh = HexahedronMesh(hex_node, hex_cell)

        dart_mesh = DartMesh.from_mesh(hex_mesh)
        dart = dart_mesh.dart
        np.testing.assert_array_equal(bm.to_numpy(dart), meshdata["dart"])


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", geo_data)
    def test_geo(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        dart = bm.from_numpy(meshdata['dart'])
        dart_mesh = DartMesh(node, dart)

        ND = dart_mesh.number_of_darts()
        NN = dart_mesh.number_of_nodes()
        NE = dart_mesh.number_of_edges()
        NF = dart_mesh.number_of_faces()
        NC = dart_mesh.number_of_cells()
        assert ND == meshdata["ND"]
        assert NN == meshdata["NN"]
        assert NE == meshdata["NE"]
        assert NF == meshdata["NF"]
        assert NC == meshdata["NC"]

        bd_dart_flag = dart_mesh.boundary_dart_flag()
        bd_dart_idx = dart_mesh.boundary_dart_index()
        np.testing.assert_array_equal(bm.to_numpy(bd_dart_flag), meshdata["bd_dart_flag"])
        np.testing.assert_array_equal(bm.to_numpy(bd_dart_idx), meshdata["bd_dart_idx"])
        bd_node_flag = dart_mesh.boundary_node_flag()
        bd_node_idx = dart_mesh.boundary_node_index()
        np.testing.assert_array_equal(bm.to_numpy(bd_node_flag), meshdata["bd_node_flag"])
        np.testing.assert_array_equal(bm.to_numpy(bd_node_idx), meshdata["bd_node_idx"])
        bd_edge_flag = dart_mesh.boundary_edge_flag()
        bd_edge_idx = dart_mesh.boundary_edge_index()
        np.testing.assert_array_equal(bm.to_numpy(bd_edge_flag), meshdata["bd_edge_flag"])
        np.testing.assert_array_equal(bm.to_numpy(bd_edge_idx), meshdata["bd_edge_idx"])
        bd_face_flag = dart_mesh.boundary_face_flag()
        bd_face_idx = dart_mesh.boundary_face_index()
        np.testing.assert_array_equal(bm.to_numpy(bd_face_flag), meshdata["bd_face_flag"])
        np.testing.assert_array_equal(bm.to_numpy(bd_face_idx), meshdata["bd_face_idx"])
        bd_cell_flag = dart_mesh.boundary_cell_flag()
        bd_cell_idx = dart_mesh.boundary_cell_index()
        np.testing.assert_array_equal(bm.to_numpy(bd_cell_flag), meshdata["bd_cell_flag"])
        np.testing.assert_array_equal(bm.to_numpy(bd_cell_idx), meshdata["bd_cell_idx"])


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", top_data)
    def test_top(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        dart = bm.from_numpy(meshdata['dart'])
        dart_mesh = DartMesh(node, dart)

        cell2face = dart_mesh.cell_to_face()
        np.testing.assert_array_equal(bm.to_numpy(cell2face[0]), meshdata["cell2face"][0])
        np.testing.assert_array_equal(bm.to_numpy(cell2face[1]), meshdata["cell2face"][1])
        cell2edge = dart_mesh.cell_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(cell2edge[0]), meshdata["cell2edge"][0])
        np.testing.assert_array_equal(bm.to_numpy(cell2edge[1]), meshdata["cell2edge"][1])
        cell2node = dart_mesh.cell_to_node()
        np.testing.assert_array_equal(bm.to_numpy(cell2node[0]), meshdata["cell2node"][0])
        np.testing.assert_array_equal(bm.to_numpy(cell2node[1]), meshdata["cell2node"][1])
        cell2cell = dart_mesh.cell_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(cell2cell[0]), meshdata["cell2cell"][0])
        np.testing.assert_array_equal(bm.to_numpy(cell2cell[1]), meshdata["cell2cell"][1])
        face2edge = dart_mesh.face_to_edge()
        np.testing.assert_array_equal(bm.to_numpy(face2edge[0]), meshdata["face2edge"][0])
        np.testing.assert_array_equal(bm.to_numpy(face2edge[1]), meshdata["face2edge"][1])
        face2node = dart_mesh.face_to_node()
        np.testing.assert_array_equal(bm.to_numpy(face2node[0]), meshdata["face2node"][0])
        np.testing.assert_array_equal(bm.to_numpy(face2node[1]), meshdata["face2node"][1])
        face2cell = dart_mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])
        edge2node = dart_mesh.edge_to_node()
        np.testing.assert_array_equal(bm.to_numpy(edge2node), meshdata["edge2node"])

        # np.testing.assert_allclose(bm.to_numpy(edge_barycenter), meshdata["edge_barycenter"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_to_vtk_and_print(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['hex_node'])
        dart = bm.from_numpy(meshdata['dart'])
        dart_mesh = DartMesh(node, dart)

        dart_mesh.nodedata = {"temperature": bm.arange(dart_mesh.number_of_nodes())}
        dart_mesh.celldata = {"pressure": bm.arange(dart_mesh.number_of_cells())}

        dart_mesh.to_vtk(f"test_dart_mesh_{backend}.vtu")

        print("\n==============================")
        print(f"Testing DartMesh with backend: {backend}")
        # dart_mesh.print()
        print(dart_mesh)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", barycenter_data)
    def test_barycenter(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        dart = bm.from_numpy(meshdata['dart'])
        dart_mesh = DartMesh(node, dart)

        edge_barycenter = dart_mesh.entity_barycenter(entityType="edge")
        face_barycenter = dart_mesh.entity_barycenter(entityType="face")
        cell_barycenter = dart_mesh.entity_barycenter(entityType="cell")
        np.testing.assert_allclose(bm.to_numpy(edge_barycenter), meshdata["edge_barycenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(face_barycenter), meshdata["face_barycenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(cell_barycenter), meshdata["cell_barycenter"], atol=1e-7)

        edge_circumcenter = dart_mesh.entity_circumcenter(entityType="edge")
        face_circumcenter = dart_mesh.entity_circumcenter(entityType="face")
        cell_circumcenter = dart_mesh.entity_circumcenter(entityType="cell")
        np.testing.assert_allclose(bm.to_numpy(edge_circumcenter), meshdata["edge_circumcenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(face_circumcenter), meshdata["face_circumcenter"], atol=1e-7)
        np.testing.assert_allclose(bm.to_numpy(cell_circumcenter), meshdata["cell_circumcenter"], atol=1e-7)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", dual_mesh_data)
    def test_barycenter(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        dart = bm.from_numpy(meshdata['dart'])
        dart_mesh = DartMesh(node, dart)

        dual_mesh = dart_mesh.dual_mesh()
        dual_mesh_node = dual_mesh.node
        np.testing.assert_allclose(bm.to_numpy(dual_mesh_node), meshdata["dual_mesh_node"], atol=1e-7)
        dual_mesh_dart = dual_mesh.dart
        np.testing.assert_array_equal(bm.to_numpy(dual_mesh_dart), meshdata["dual_mesh_dart"])

        dual_mesh.to_vtk(f"test_dual_dart_mesh_{backend}.vtu")



if __name__ == "__main__":
    pytest.main(["./test_dart_mesh.py", "-k", "test_init_and_from_mesh"])
