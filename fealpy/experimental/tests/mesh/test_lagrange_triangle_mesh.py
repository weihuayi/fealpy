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
    def test_init(self, data, backend):
        bm.set_backend(backend)

if __name__ == "__main__":
    #a = LagrangeTestTriangleMeshInterfaces()
    #a.test_grad_shape_function(grad_shape_function_data[0], 'pytorch')
    pytest.main(["./test_lagrange_triangle_mesh.py"])
