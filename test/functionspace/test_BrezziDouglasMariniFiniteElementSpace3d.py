
import numpy as np
import pytest

from fealpy.functionspace.brezzi_douglas_marini_fe_space_3d import BDMDof
from fealpy.functionspace.brezzi_douglas_marini_fe_space_3d import BDMFiniteElementSpace3d
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh

from BrezziDouglasMariniFiniteElementSpace3d_data import *

class TestBDMDof:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  3
        a = BDMDof(mesh,p)
        assert a.number_of_local_dofs() == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

class TestBDMFiniteElementSpace3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis_vector(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  2
        a = BDMFiniteElementSpace3d(mesh,p)
        
        basis_vector = a.basis_vector()
        np.testing.assert_allclose(bm.to_numpy(basis_vector), meshdata["basis_vector"],1e-8)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = BDMFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.3,0.1,0.5],
                 [0.2,0.2,0.2,0.4]])
        basis = a.basis(bcs)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-6)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_div_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = BDMFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.3,0.1,0.5],
                 [0.2,0.2,0.2,0.4]],dtype=bm.float64)
        div_basis = a.div_basis(bcs)
        np.testing.assert_allclose(bm.to_numpy(div_basis), meshdata["div_basis"],1e-7)


if __name__ == "__main__":
    #test = TestBDMDof()
    #test.test_cell_to_dof(init_data[0],'numpy')
    test = TestBDMFiniteElementSpace3d()
    #test.test_basis_vector(init_data[0],'pytorch')
    #test.test_basis(init_data[0],'pytorch')
    test.test_div_basis(init_data[0],'numpy')