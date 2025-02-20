import ipdb
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import CmConformingFESpace2d 
from fealpy.fem import BilinearForm 
from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator

from mthlaplace_integrator_data import *


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

    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    @pytest.mark.parametrize("data", mass)
    def test_mass_integrator(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1], 1, 1)
        node = mesh.entity('node')
        isCornerNode = bm.zeros(len(node),dtype=bm.bool)
        for n in bm.array([[0,0],[1,0],[0,1],[1,1]], dtype=bm.float64):
            isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)

        space = CmConformingFESpace2d(mesh, 5, 1, isCornerNode)
        bform = BilinearForm(space)
        integrator = MthLaplaceIntegrator(m=0, coef=1, q=9)
        FM = integrator.assembly(space)
        bform.add_integrator(integrator)

        M = bform.assembly()
        np.testing.assert_allclose(M.toarray(), data['M1'], atol=1e-14)
 

    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    @pytest.mark.parametrize("data", grad_m)
    def test_assembly_without_numerical_mlaplace_integrator(self, backend, data): 
        from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
        from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d 
        from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator

        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], nx=1, ny=1, nz=1)
        node = mesh.entity('node')
        isCornerNode = bm.zeros(len(node),dtype=bm.bool)
        for n in bm.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1],[0,1,1],[0,0,1],[1,0,1]], dtype=bm.float64):
            isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace3d(mesh, p=9, m=1, isCornerNode=isCornerNode)
         
        bform = BilinearForm(space)
        integrator = MthLaplaceIntegrator(m=2, coef=1, q=14, method='without_numerical_integration')
        bform.add_integrator(integrator)
        M = bform.assembly()

        bform = BilinearForm(space)
        integrator = MthLaplaceIntegrator(m=2, coef=1, q=14)
        bform.add_integrator(integrator)
        M1 = bform.assembly()
 

        np.testing.assert_allclose(bm.to_numpy(M.toarray()), bm.to_numpy(M1.toarray()), atol=1e-14)
        print('11111')

if __name__ == "__main__":
    #pytest.main(['test_grad_m_integrator.py', "-q", "-k","test_grad_m_integrator", "-s"])
    t = TestgradmIntegrator()
    t.test_mass_integrator('numpy', mass[0])

    #t.test_grad_m_integrator('numpy', grad_m[0])
    #t.test_assembly_without_numerical_mlaplace_integrator('numpy', grad_m[0])
    #t.test_assembly_without_numerical_mlaplace_integrator('pytorch', grad_m[0])
