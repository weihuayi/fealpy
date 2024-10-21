
import ipdb
import matplotlib.pyplot as plt
import pytest
import numpy as np
import jax.numpy as jnp
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.backend import backend_manager as bm
from node_mesh_data import *

class TestNodeMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_number_of_node(self, meshdata, backend):
        bm.set_backend(backend)
        nodes = bm.from_numpy(meshdata["node"])
        node_mesh =  NodeMesh(nodes)
        num = node_mesh.number_of_nodes()
        assert num == meshdata["num"]
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_geo_dimension(self, meshdata, backend):
        bm.set_backend(backend)
        nodes = bm.from_numpy(meshdata["node"])
        node_mesh =  NodeMesh(nodes)
        geo_dim = node_mesh.geo_dimension()
        assert geo_dim == meshdata["geo_dim"]
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_top_dimension(self, meshdata, backend):
        bm.set_backend(backend)
        nodes = bm.from_numpy(meshdata["node"])
        node_mesh =  NodeMesh(nodes)
        top = node_mesh.top_dimension()
        assert top == meshdata["top"]
    
    @pytest.mark.parametrize("backend", ['numpy','jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_neighbors(self, meshdata, backend):
        bm.set_backend(backend)
        nodes = bm.from_numpy(meshdata["node_box"])
        node_mesh = NodeMesh(nodes)
        index, indptr = node_mesh.neighbors(meshdata["box_size"], meshdata["cutoff"])
        assert jnp.array_equal(index, meshdata["index"])
        assert np.array_equal(index, meshdata["index"])
        assert jnp.array_equal(indptr, meshdata["indptr"])
        assert np.array_equal(indptr, meshdata["indptr"])

    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_from_tgv_domain(self, meshdata, backend):
        bm.set_backend(backend)
        box_size = bm.tensor([meshdata["box_size"], meshdata["box_size"]])
        node = NodeMesh.from_tgv_domain(box_size)
        tgv_num = node.number_of_node()
        assert tgv_num == meshdata["tgv_num"]
        
    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_from_heat_transfer_domain(self, meshdata, backend):
        bm.set_backend(backend)
        nodemesh = NodeMesh.from_heat_transfer_domain()
        node = nodemesh.node
        ht_num = node.shape[0]
        assert ht_num == meshdata["ht_num"]

    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_from_four_heat_transfer_domain(self, meshdata, backend):
        bm.set_backend(backend)
        nodemesh = NodeMesh.from_four_heat_transfer_domain()
        node = nodemesh.node
        fht_num = node.shape[0]
        assert fht_num == meshdata["fht_num"]
    
    @pytest.mark.parametrize("backend", ['numpy', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_from_long_rectangular_cavity_domain(self, meshdata, backend):
        bm.set_backend(backend)
        node_set = NodeMesh.from_long_rectangular_cavity_domain()

        positions = node_set.nodedata["position"]
        tags = node_set.nodedata["tag"]
        lrc_num = positions.shape[0]
        assert lrc_num == meshdata["lrc_num"]

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("meshdata", mesh_data)
    def test_from_dam_break_domain(self, meshdata, backend):
        bm.set_backend(backend)
        node_set = NodeMesh.from_dam_break_domain()
        db_num = node_set.number_of_node()
        assert db_num == meshdata["db_num"]

if __name__ == "__main__":
    pytest.main(["./test_node_mesh.py"])

