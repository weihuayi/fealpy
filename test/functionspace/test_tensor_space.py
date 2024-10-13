
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace,TensorFunctionSpace
from fealpy.typing import TensorLike, Size, _S

from tensor_space_data import *


class TestTensorFunctionSpace:  

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh)
    def test_top(self, backend, data): 
        bm.set_backend(backend)
        
        mesh_tri = TriangleMesh.from_box([0,1,0,1],2,2)
        qf = mesh_tri.quadrature_formula(2, 'cell')
        
        bcs, ws = qf.get_quadrature_points_and_weights()
        space_tri = LagrangeFESpace(mesh_tri, p=2, ctype='C')
        GD_tri = space_tri.geo_dimension()  

        tensor_space = TensorFunctionSpace(space_tri, shape=(GD_tri, -1))
        tdofnumel = tensor_space.dof_numel 
        tGD = tensor_space.dof_ndim 
        tld = space_tri.number_of_local_dofs()
        tcell2dof = tensor_space.cell_to_dof()
        tface2dof = tensor_space.face_to_dof() 
        tphi = tensor_space.basis(bcs, index=_S)
        tgrad_phi = tensor_space.grad_basis(bcs, index=_S, variable='x')
        

        np.testing.assert_array_equal(tdofnumel, data["tdofnumel"], 
                                     err_msg=f" `tdofnumel` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(tGD, data["GD"], 
                                     err_msg=f" `number_of_global_dofs` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(tld, data["ld"], 
                                    err_msg=f" `number_of_local_dofs` function is not equal to real result in backend {backend}")
#        np.testing.assert_array_equal(tface2dof, data["tface2dof"], 
#                                     err_msg=f" `face_to_dof` function is not equal to real result in backend {backend}")
#        np.testing.assert_array_equal(tcell2dof, data["tcell2dof"], 
#                                     err_msg=f" `cell_to_dof` function is not equal to real result in backend {backend}")
        np.testing.assert_almost_equal(bm.to_numpy(tphi),  data["tphi"], decimal=7, 
                                     err_msg=f" `basis` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(tgrad_phi, data["tgrad_phi"], 
                                     err_msg=f" `grad_basis` function is not equal to real result in backend {backend}")
       



if __name__ == "__main__":
    pytest.main(['test_tensor_space.py', "-q"])
