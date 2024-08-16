import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.experimental.tests.functionspace.first_nedelec_fe_space_3d_data import*
from fealpy.experimental.functionspace.first_nedelec_fe_space_3d import FirstNedelecDof3d
from fealpy.experimental.functionspace.first_nedelec_fe_space_3d import FirstNedelecFiniteElementSpace3d


class TestFirstNedelecDof3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  3
        a = FirstNedelecDof3d(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])


class TestFirstNedelecFiniteElementSpace3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.2,0.1,0.6],
                         [0.5,0.3,0.1,0.1]],dtype=mesh.ftype)

        basis = a.basis(bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-15)



if __name__ == "__main__":
    # test = TestFirstNedelecDof3d()
    # test.test_cell_to_dof(init_data[0],'pytorch')
    test = TestFirstNedelecFiniteElementSpace3d()
    # test.test_basis(init_data[0],"pytorch")