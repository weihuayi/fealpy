import ipdb
import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.functionspace import HuZhangFESpace2D
from fealpy.experimental.tests.functionspace.huzhang_fe_space_2d_data import *

class TestHuZhangFiniteElementSpace2D:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_top(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = HuZhangFESpace2D(mesh, 1)
        ldofs = space.number_of_local_dofs()
        gdofs = space.number_of_global_dofs()
        TD = space.top_dimension()
        GD = space.geo_dimension()

if __name__ == "__main__":
    pytest.main(['test_huzhang_fe_space_2d.py', "-q"])
