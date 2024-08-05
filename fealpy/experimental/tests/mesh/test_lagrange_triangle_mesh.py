import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.experimental.tests.mesh.lagrange_triangle_mesh_data import *

class LagrangeTestTriangleMeshInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend, p):
        bm.set_backend(backend)
 
        p = bm.from_numpy(data['p'])
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])
        surface = bm.from_numpy(data['surface'])

        mesh = LagrangeTriangleMesh(node, cell, p=p, surface=surface)
        
        assert mesh.number_of_nodes() == data["NN"] 
        assert mesh.number_of_cells() == data["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_allclose(bm.to_numpy(face2cell), data["face2cell"], atol=1e-14)   

if __name__ == "__main__":
    #a = LagrangeTestTriangleMeshInterfaces()
    #a.test_grad_shape_function(grad_shape_function_data[0], 'pytorch')
    pytest.main(["./test_lagrange_triangle_mesh.py"])
