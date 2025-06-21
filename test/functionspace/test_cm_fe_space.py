from fealpy.backend import backend_manager as bm
import numpy as np
import pytest
import sympy as sp
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace.cm_conforming_fe_space import CmConformingFESpace2d
from cm_fe_space_data import *

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

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_matrix(self, backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node),dtype=np.bool)
        for n in np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace2d(mesh, 5, 1, isCornerNode) 

        x = sp.symbols('x')
        y = sp.symbols('y')
        f = x**2*y**3
        from fealpy.experimental.pde.biharmonic_triharmonic_2d import get_flist
        get_flist = get_flist(f)

        from fealpy.experimental.fem import BilinearForm
        from fealpy.experimental.fem import LinearForm
        from fealpy.experimental.fem.scalar_mlaplace_source_integrator import ScalarMLaplaceSourceIntegrator
        from fealpy.experimental.fem.mthlaplace_integrator import MthLaplaceIntegrator
        lform = LinearForm(space)
        lform.add_integrator(ScalarMLaplaceSourceIntegrator(2, get_flist[2], q=10))
        gF = lform.assembly()

        bform = BilinearForm(space)                                                 
        bform.add_integrator(MthLaplaceIntegrator(m=2, coef=1,q=10))                     
        A = bform.assembly()  

        fi = space.interpolation(get_flist)
        x = fi[:]
        aa = A@x
        print(bm.abs(aa-gF).max())
        #np.testing.assert_allclose(aa, gF, atol=1e-14,rtol=0)

    def test_isConerNode(self):
        mesh = TriangleMesh.from_box([0,1,0,1],2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node),dtype=np.bool)
        for n in np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace2d(mesh, 5, 1, isCornerNode) 
        space.isCornerNode()


if __name__=="__main__":
    t = TestCmfespace2d()
    #t.test_matrix("pytorch")
    t.test_isConerNode()

