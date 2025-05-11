
import numpy as np
import pytest

from fealpy.functionspace.brezzi_douglas_marini_fe_space_2d import BDMDof
from fealpy.functionspace.brezzi_douglas_marini_fe_space_2d import BDMFiniteElementSpace2d
from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh

from BrezziDouglasMariniFiniteElementSpace2d_data import *

class TestBDMDof:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  3
        a = BDMDof(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs() == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

class TestBDMFiniteElementSpace2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis_vector(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  3
        a = BDMFiniteElementSpace2d(mesh,p)
        
        basis_vector = a.basis_vector()
        np.testing.assert_allclose(bm.to_numpy(basis_vector), meshdata["basis_vector"],1e-8)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  3
        a = BDMFiniteElementSpace2d(mesh,p)
        bcs = bm.array([[0.1,0.2,0.7],
                [0.5,0.2,0.3],
                [0.1,0.4,0.5]])
        basis = a.basis(bcs)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-6)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_div_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  3
        a = BDMFiniteElementSpace2d(mesh,p)
        bcs = bm.array([[0.1,0.2,0.7],
                [0.5,0.2,0.3],
                [0.1,0.4,0.5]],dtype=bm.float64)
        div_basis = a.div_basis(bcs)
        np.testing.assert_allclose(bm.to_numpy(div_basis), meshdata["div_basis"],1e-7)


if __name__ == "__main__":
    # test = TestBDMDof()
    # test.test_cell_to_dof(init_data[0],'pytorch')
    test = TestBDMFiniteElementSpace2d()
    #test.test_basis_vector(init_data[0],'numpy')
    #test.test_basis(init_data[0],'numpy')
    test.test_div_basis(init_data[0],'pytorch')