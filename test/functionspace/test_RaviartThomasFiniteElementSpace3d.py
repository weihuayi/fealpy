import numpy as np
import pytest

from fealpy.functionspace.RaviartThomasFiniteElementSpace3d import RTDof3d
from fealpy.functionspace.RaviartThomasFiniteElementSpace3d import RTFiniteElementSpace3d
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh

from RaviartThomasFiniteElementSpace3d_data import*

class TestFirstNedelecDof3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  2
        a = RTDof3d(mesh,p)
        assert a.number_of_local_dofs("face") == meshdata["fdof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

class TestRTFiniteElementSpace3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = RTFiniteElementSpace3d(mesh,p)
        bcs = bm.array([[0.1,0.2,0.3,0.4],
                [0.5,0.1,0.3,0.1]])
        basis = a.basis(bcs)

        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_div_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = RTFiniteElementSpace3d(mesh,p)
        bcs = bm.array([[0.1,0.2,0.3,0.4],
                [0.5,0.1,0.3,0.1]])
        div_basis = a.div_basis(bcs)

        np.testing.assert_allclose(bm.to_numpy(div_basis), meshdata["div_basis"],1e-14)


if __name__ == "__main__":
    # test = TestFirstNedelecDof3d()
    # test.test_cell_to_dof(init_data[0],'numpy')
    test = TestRTFiniteElementSpace3d()
    #test.test_basis(init_data[0],'numpy')
    test.test_div_basis(init_data[0],'numpy')