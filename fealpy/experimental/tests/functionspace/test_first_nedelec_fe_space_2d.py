import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.tests.functionspace.fist_nedelec_fe_space_2d_data import*
from fealpy.experimental.functionspace.first_nedelec_fe_space_2d import FirstNedelecDof2d


class TestFirstNedelecDof2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  4
        a = FirstNedelecDof2d(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dof() == meshdata["gdof"]
        
        cell2dof = a.cell2dof

        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])


 


if __name__ == "__main__":
   pytest.main(["test_first_nedelec_fe_space_2d.py"])
