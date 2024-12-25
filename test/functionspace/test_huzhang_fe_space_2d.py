
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import HuZhangFESpace2D

from huzhang_fe_space_data_2d import *


class TestHuZhangFiniteElementSpace2D:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_top(self, backend): 
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = HuZhangFESpace2D(mesh, 1)
        ldofs = space.number_of_local_dofs()
        gdofs = space.number_of_global_dofs()
        TD = space.top_dimension()
        GD = space.geo_dimension()

if __name__ == "__main__":
    pytest.main(['test_huzhang_fe_space_2d.py', "-q"])
