import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.tests.functionspace.lagrange_fe_space_data import *

class TestLagrangeFiniteElementSpace:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_top(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        ldofs = space.number_of_local_dofs()
        gdofs = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        face2dof = space.face_to_dof()
        edge2dof = space.edge_to_dof()
        #np.testing.assert_array_equal(ldofs, data["number_of_local_dofs"], 
        #                             err_msg=f" `number_of_local_dofs` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(gdofs, data["number_of_global_dofs"], 
                                     err_msg=f" `number_of_global_dofs` function is not equal to real result in backend {backend}")
        #np.testing.assert_array_equal(cell2dof, data["cell_to_dof"], 
        #                             err_msg=f" `cell_to_dof` function is not equal to real result in backend {backend}")
        #np.testing.assert_array_equal(face2dof, data["face_to_dof"], 
        #                             err_msg=f" `face_to_dof` function is not equal to real result in backend {backend}")
        #np.testing.assert_array_equal(edge2dof, data["edge_to_dof"], 
        #                             err_msg=f" `edge_to_dof` function is not equal to real result in backend {backend}")

if __name__ == "__main__":
    #pytest.main(['test_lagrange_fe_space.py', "-q"])
    bm.set_backend('pytorch') 
    mesh = TriangleMesh.from_box([0,1,0,1],1,1)
    space = LagrangeFESpace(mesh, 2)
    ldofs = space.number_of_local_dofs()
    gdofs = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    face2dof = space.face_to_dof()
    edge2dof = space.edge_to_dof()
