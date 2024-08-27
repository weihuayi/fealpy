import ipdb
import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import CmConformingFESpace2d 
from fealpy.experimental.fem import BilinearForm 
from fealpy.experimental.fem.mthlaplace_integrator import MthLaplaceIntegrator
from fealpy.experimental.tests.fem.mthlaplace_integrator_data import *
class TestgradmIntegrator:
    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    @pytest.mark.parametrize("data", grad_m)
    def test_grad_m_integrator(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1], 2, 2)
        node = mesh.entity('node')
        isCornerNode = bm.zeros(len(node),dtype=bm.bool)
        for n in bm.array([[0,0],[1,0],[0,1],[1,1]], dtype=bm.float64):
            isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)

        p = data['p']
        space = CmConformingFESpace2d(mesh, p, data["spacem"], isCornerNode)
        bform = BilinearForm(space)
        integrator = MthLaplaceIntegrator(m=data["equationm"], coef=1, q=p+4)
        FM = integrator.assembly(space)
        bform.add_integrator(integrator)

        M = bform.assembly()
 
        np.testing.assert_allclose(bm.to_numpy(FM), data["FM"], atol=1e-14)
        #np.testing.assert_allclose(bm.to_numpy(M.toarray()), data["M"], atol=1e-14)

if __name__ == "__main__":
    pytest.main(['test_grad_m_integrator.py', "-q", "-k","test_grad_m_integrator", "-s"])
