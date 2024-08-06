import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh

from fealpy.experimental.tests.functionspace.bernstein_fe_space_data import*
from fealpy.experimental.functionspace.bernstein_fe_space import BernsteinFESpace


class TestBernsteinFESpace:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_init_(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        assert a.number_of_local_dofs("cell") == meshdata["ldof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        
        interpolation = a.interpolation_points()
        c2d = a.cell_to_dof()
        f2d = a.face_to_dof()
        bdof = a.is_boundary_dof()

        np.testing.assert_array_equal(bm.to_numpy(interpolation), meshdata["interpolation"])
        np.testing.assert_array_equal(bm.to_numpy(c2d), meshdata["c2d"])
        np.testing.assert_array_equal(bm.to_numpy(f2d), meshdata["f2d"])
        np.testing.assert_array_equal(bm.to_numpy(bdof), meshdata["bdof"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        basis = a.basis(bcs=bcs,p=1)
        np.testing.assert_array_equal(bm.to_numpy(basis), meshdata["basis"])
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_value(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        uh = bm.tensor([1,2,3,4],dtype=mesh.ftype)
        value = a.value(uh=uh,bcs=bcs)
        np.testing.assert_array_equal(bm.to_numpy(value), meshdata["value"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_l_to_b(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        ltob = a.lagrange_to_bernstein()
        np.testing.assert_array_equal(bm.to_numpy(ltob), meshdata["ltob"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_b_to_l(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        btol = a.bernstein_to_lagrange()
        np.testing.assert_array_equal(bm.to_numpy(btol), meshdata["btol"])




if __name__ == "__main__":

    test = TestBernsteinFESpace()
    # test.test_init_(init_data[0],"pytorch")
    # test.test_basis(init_data[0],"pytorch")
    # test.test_grad_basis(init_data[0],"pytorch")
    # test.test_value(init_data[0],"pytorch")
    # test.test_l_to_b(init_data[0],"pytorch")
    # test.test_b_to_l(init_data[0],"pytorch")