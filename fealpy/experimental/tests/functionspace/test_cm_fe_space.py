from ..backend import backend_manager as bm
import numpy as np
import pytest
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.functionspace.cm_conforming_fe_space import CmConformingFESpace2d

class TestCmfespace2d:
    @pytest.mark.parametrize("backend", ["numpy", "pytorch", "jax"])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        np.testing.assert_allclose(bm.to_numpy(n), data["fn"], atol=1e-14)

        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node),dtype=np.bool)
        for n in np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace2d(mesh, 5, 1, isCornerNode) 
