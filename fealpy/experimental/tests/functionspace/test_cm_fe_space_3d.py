import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.experimental.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d
from fealpy.experimental.tests.functionspace.cm_fe_space_3d_data import *

class TestCmfespace3d:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", get_dof_index )
    def test_get_dof_index(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        get_dof_index = space.get_dof_index()


