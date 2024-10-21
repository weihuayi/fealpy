import ipdb
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d 
from fealpy.fem import BilinearForm 
from fealpy.fem.mlaplace_bernstein_integrator import MLaplaceBernsteinIntegrator

class TestMLaplaceBernsteinIntegrator:
    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    #@pytest.mark.parametrize("data", grad_m)
    def test_mlaplace_berstein_integrator(self, backend, data): 
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)

        bform = BilinearForm(space)                                                 
        bform.add_integrator(MLaplaceBernsteinIntegrator(m=2, coef=1,q=14))                     
        A = bform.assembly()  


       

if __name__ == "__main__":
    t = TestMLaplaceBernsteinIntegrator()
    #t.test_mlaplace_berstein_integrator('pytorch', None)
    t.test_mlaplace_berstein_integrator('numpy', None)
