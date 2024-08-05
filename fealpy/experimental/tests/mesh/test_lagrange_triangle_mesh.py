import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.experimental.tests.mesh.lagrange_triangle_mesh_data import *

class TestLagrangeTriangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", from_triangle_mesh_data)
    def test_from_triangle_mesh(self, data, backend):
        bm.set_backend(backend)

        p = data['p']
        tmesh = data['mesh']
        surface = data['surface']

        mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh, p=p, surface=surface)
        
        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_edges() == data["NE"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        edge2cell = mesh.edge_to_cell()
        print(data['face2cell']-edge2cell)
        np.testing.assert_allclose(bm.to_numpy(edge2cell), data["face2cell"], atol=1e-14)   

if __name__ == "__main__":
    a = TestLagrangeTriangleMeshInterfaces()
    a.test_from_triangle_mesh(from_triangle_mesh_data[0], 'numpy')
    #pytest.main(["./test_lagrange_triangle_mesh.py"])
