
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from lagrange_fe_space_data import *


class TestLagrangeFiniteElementSpace:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_top(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        ldofs = space.number_of_local_dofs()
        gdofs = space.number_of_global_dofs()
        TD = space.top_dimension()
        GD = space.geo_dimension()
        isBDof = space.is_boundary_dof()
        cell2dof = space.cell_to_dof()
        face2dof = space.face_to_dof()
        edge2dof = space.edge_to_dof()
        np.testing.assert_array_equal(ldofs, data["number_of_local_dofs"], 
                                     err_msg=f" `number_of_local_dofs` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(gdofs, data["number_of_global_dofs"], 
                                     err_msg=f" `number_of_global_dofs` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(cell2dof, data["cell_to_dof"], 
                                    err_msg=f" `cell_to_dof` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(face2dof, data["face_to_dof"], 
                                     err_msg=f" `face_to_dof` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(edge2dof, data["edge_to_dof"], 
                                     err_msg=f" `edge_to_dof` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(GD, data["geo_dimension"], 
                                     err_msg=f" `geo_dimension` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(TD, data["top_dimension"], 
                                     err_msg=f" `top_dimension` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(isBDof, data["is_boundary_dof"], 
                                             err_msg=f" `edge_to_dof` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("backend", ['numpy','pytorch'])    
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_basis(self, backend, data): 
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        bcs = bm.array(data["bcs"])
        phi = space.basis(bcs)
        gphi = space.grad_basis(bcs)
        np.testing.assert_array_almost_equal(phi, data["basis"], 
                                     err_msg=f" `basis` function is not equal to real result in backend {backend}")
        np.testing.assert_array_almost_equal(gphi, data["grad_basis"], 
                                     err_msg=f" `grad_basis` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_value(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        bcs = bm.array(data["bcs"])
        
        def test_fun(point): 
            x = point[..., 0]
            y = point[..., 1]
            result = bm.sin(x) * bm.sin(y)
            return result
        
        uh = space.interpolate(test_fun)
        value = space.value(uh, bcs)
        grad_value = space.grad_value(uh, bcs)
        np.testing.assert_almost_equal(data["interpolate"], bm.to_numpy(uh),decimal=7, 
                                     err_msg=f" `interpolate` function is not equal to real result in backend {backend}")
        np.testing.assert_almost_equal(data["value"], bm.to_numpy(value),decimal=7, 
                                     err_msg=f" `value` function is not equal to real result in backend {backend}")
        np.testing.assert_almost_equal(data["grad_value"], bm.to_numpy(grad_value),decimal=7, 
                                     err_msg=f" `grad_value` function is not equal to real result in backend {backend}")


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_boundary_interpolate(self, backend, data): 
        bm.set_backend(backend)
        def solution(p):
            x = p[..., 0]
            y = p[..., 1]
            return bm.sin(bm.pi*x) * bm.sin(bm.pi*y)
        
        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        space = LagrangeFESpace(mesh, 2)
        uh = np.zeros(space.number_of_global_dofs())
        b = space.boundary_interpolate(gD=solution, uh=uh)
        np.testing.assert_array_almost_equal(uh, data["uh"], 
                                     err_msg=f" `uh` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(b, data["isDDof"], 
                                     err_msg=f" `b` function is not equal to real result in backend {backend}")





if __name__ == "__main__":
    #pytest.main(['test_lagrange_fe_space.py', "-q", "-k","test_basis", "-s"])
    pytest.main(['test_lagrange_fe_space.py', "-q"])
