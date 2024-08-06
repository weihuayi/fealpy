import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.experimental.tests.mesh.lagrange_triangle_mesh_data import *

class TestLagrangeTriangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", init_data)
    def test_init_mesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        node = data['node']
        cell = data['cell']
        surface = data['surface']

        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)

        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        #face2cell = mesh.face_to_cell()
        #np.testing.assert_allclose(bm.to_numpy(face2cell), data["face2cell"], atol=1e-14)   

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", from_triangle_mesh_data)
    def test_from_triangle_mesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        mesh = data['mesh']
        surface = data['surface']

        lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=p, surface=surface)

        assert lmesh.number_of_nodes() == data["NN"] 
        assert lmesh.number_of_edges() == data["NE"] 
        assert lmesh.number_of_cells() == data["NC"] 
        
        #face2cell = lmesh.face_to_cell()
        #np.testing.assert_allclose(bm.to_numpy(face2cell), data["face2cell"], atol=1e-14)   

if __name__ == "__main__":
    a = TestLagrangeTriangleMeshInterfaces()
    a.test_init_mesh(init_data[0], 'numpy')
    #a.test_from_triangle_mesh(from_triangle_mesh_data[0], 'numpy')
    #pytest.main(["./test_lagrange_triangle_mesh.py"])
