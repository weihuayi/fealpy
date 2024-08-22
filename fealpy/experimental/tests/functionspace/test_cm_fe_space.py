from fealpy.experimental.backend import backend_manager as bm
import numpy as np
import pytest
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.functionspace.cm_conforming_fe_space import CmConformingFESpace2d
from fealpy.experimental.tests.functionspace.cm_fe_space_data import *

class TestCmfespace2d:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", cell_to_dof)
    def test_cell_to_dof(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node),dtype=np.bool)
        for n in np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace2d(mesh, 8, 1, isCornerNode) 
        n2d  = space.cell_to_dof()

        np.testing.assert_equal(bm.to_numpy(n2d), data["c2d"])
        #np.testing.assert_allclose(bm.to_numpy(n2d), data["c2d"], atol=1e-14)
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", is_boundary_dof)
    def test_is_boundary_dof(self, data, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node),dtype=np.bool)
        for n in np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace2d(mesh, 8, 1, isCornerNode) 

        isBdDof = space.is_boundary_dof()
        np.testing.assert_equal(bm.to_numpy(isBdDof), data["isBdDof"])
