import numpy as np
from fealpy.functionspace import ConformingScalarVESpace2d
import numpy as np
import matplotlib.pyplot as plt
import pytest
from scipy.sparse import csr_matrix
from fealpy.backend import backend_manager as bm

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d

from  conforming_scalar_ve_space2d_data import *

class TestConformingScalarVESpace2d():
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",dof)
    def test_dof(self, backend, data):
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        space = ConformingScalarVESpace2d(mesh, p=3)
        edge_to_dof = space.edge_to_dof()
        cell_to_dof, cell_to_dof_location = space.cell_to_dof()
        def f(p):
            x = p[...,0]
            y = p[..., 1]
            flag = np.abs(x) < 1e-10
            return flag
        is_boundary_dof = space.is_boundary_dof(f)
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        assert gdof == data["gdof"]
        np.testing.assert_equal(edge_to_dof, data["edge_to_dof"])
        np.testing.assert_equal(cell_to_dof, data["cell_to_dof"])
        np.testing.assert_equal(cell_to_dof_location,
                                data["cell_to_dof_location"])

        #np.testing.assert_equal(multi_index_matrix, data["multi_index_matrix"])
if __name__ == "__main__":
    t = TestConformingScalarVESpace2d()
    t.test_dof('pytorch', dof[0])
    t.test_dof('numpy', dof[0])

