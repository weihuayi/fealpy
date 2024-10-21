
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace.bernstein_fe_space import BernsteinFESpace

from bernstein_fe_space_data import *


class TestBernsteinFESpace:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
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

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch','jax'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = BernsteinFESpace(mesh,p)
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        basis = a.basis(bcs=bcs,p=1)
        np.testing.assert_array_equal(bm.to_numpy(basis), meshdata["basis"].swapaxes(0,1))
        
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
        print(value)
        np.testing.assert_array_equal(bm.to_numpy(value), meshdata["value"].swapaxes(0,1))

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


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_hess_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        a = BernsteinFESpace(mesh,p)
        hess_basis = a.hess_basis(bcs=bcs)
        np.testing.assert_array_equal(bm.to_numpy(hess_basis), meshdata["hess_basis"].swapaxes(0,1))

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_grad_m_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        a = BernsteinFESpace(mesh,p)
        m=1
        grad_m_basis = a.grad_m_basis(bcs=bcs,m=m)
        np.testing.assert_allclose(bm.to_numpy(grad_m_basis), meshdata["gradmbasis"].swapaxes(0,1),1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_interpolate(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        a = BernsteinFESpace(mesh,p)
        def fun(p):
            x=p[...,0]
            y=p[...,1]
            return 2*bm.sin(x)+bm.cos(y)
        a = BernsteinFESpace(mesh,p)
        interpolate = a.interpolate(u = fun)

        np.testing.assert_allclose(bm.to_numpy(interpolate), meshdata["interpolate"],1e-15)
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_hessian_value(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        node = mesh.entity('node')
        node[3] + bm.array([-0.2, 0.5],dtype=mesh.ftype)
        mesh.node = node

        p =  2
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        a = BernsteinFESpace(mesh,p)
        uh = bm.array([0.8430878444388449, 0.279405787552195 , 0.3932471419048877,
       0.5433132870312605, 0.1656913889905416, 0.7326699516750773,
       0.3567817609053885, 0.8039158020389613, 0.1809355630744111],dtype=mesh.ftype)
        hessvalue = a.hessian_value(uh=uh,bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(hessvalue), meshdata["hessvalue"].swapaxes(0,1),1e-15)
   
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_grad_m_value(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        node = mesh.entity('node')
        node[3] + bm.array([-0.2, 0.5],dtype=mesh.ftype)
        mesh.node = node

        p =  2
        bcs = bm.tensor([[0.2,0.3,0.5],
                [0.1,0.1,0.8]],dtype=mesh.ftype)
        a = BernsteinFESpace(mesh,p)
        uh = bm.array([0.7144804355262809, 0.7650298689135151, 0.1210423628784636,
       0.8781921383761604, 0.7534806230528277, 0.2362274035570283,
       0.6538302387489031, 0.3537586892727872, 0.9353446746437852],dtype=mesh.ftype)
        m=1
        mvalue = a.grad_m_value(uh=uh,bcs=bcs,m=1)

        np.testing.assert_allclose(bm.to_numpy(mvalue), meshdata["mvalue"].swapaxes(0,1),1e-15)



if __name__ == "__main__":
    test = TestBernsteinFESpace()
    test.test_basis(init_data[0],'numpy')
    # pytest.main()
