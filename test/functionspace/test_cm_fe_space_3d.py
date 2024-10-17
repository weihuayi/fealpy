import pytest
import numpy as np
from fealpy.decorator import barycentric,cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.quadrature.stroud_quadrature import StroudQuadrature 
from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d
from cm_fe_space_3d_data import *
from fealpy.pde.biharmonic_triharmonic_3d import get_flist
import sympy as sp
import ipdb

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
        get_dof_index = space.dof_index["all"]        
        np.testing.assert_array_equal(get_dof_index, data["dof_index"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", number_of_internal_global_dofs)
    def test_number_of_internal_dofs(self, data, backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        gdof = space.number_of_global_dofs()
        idof = space.number_of_internal_dofs(data["etype"])
        #np.testing.assert_array_equal(ldof, data["ldof"])
        assert idof == data["ldof"]
        assert gdof == data["gdof"]


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", nefc_to_internal_dof)
    def test_nefc_to_internal_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        n2d = space.node_to_dof()
        e2id = space.edge_to_internal_dof()
        f2id = space.face_to_internal_dof()
        c2id = space.cell_to_internal_dof()
        np.testing.assert_array_equal(n2d, data["node"])
        np.testing.assert_array_equal(e2id, data["edge"])
        np.testing.assert_array_equal(f2id, data["face"])
        np.testing.assert_array_equal(c2id, data["cell"])


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", cell2dof)
    def test_cell_to_dofs(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],3,3,3)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)

        c2d = space.cell_to_dof()[data["cellnum"]]
        np.testing.assert_allclose(c2d, data["cell2dof"])

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", coefficient_matrix)
    def test_coefficient_matrix(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        coefficient_matrix = space.coefficient_matrix() #(6, 364, 364)

        np.testing.assert_allclose(coefficient_matrix[0, 180], data["cell0"], atol=1e-14)
        np.testing.assert_allclose(coefficient_matrix[1, 256], data["cell1"], atol=1e-14)
        np.testing.assert_allclose(coefficient_matrix[2, 200], data["cell2"], atol=1e-14)
        np.testing.assert_allclose(coefficient_matrix[3, 20], data["cell3"], atol=1e-14)
        np.testing.assert_allclose(coefficient_matrix[4, 105], data["cell4"], atol=1e-14)
        np.testing.assert_allclose(coefficient_matrix[5, 78], data["cell5"], atol=1e-14)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", basis)
    def test_basis(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        bcs = mesh.multi_index_matrix(p=4, etype=3)/4
        #qf = StroudQuadrature(3, 7)
        #bcs, ws = qf.get_points_and_weights()
        basis = space.basis(bcs)
 
        np.testing.assert_allclose(basis[0, 27], data["cell0"], atol=1e-14)
        np.testing.assert_allclose(basis[1, 20], data["cell1"], atol=1e-14)
        np.testing.assert_allclose(basis[2, 5],  data["cell2"], atol=1e-14)
        np.testing.assert_allclose(basis[3, 32], data["cell3"], atol=1e-14)
        np.testing.assert_allclose(basis[4, 11], data["cell4"], atol=1e-14)
        np.testing.assert_allclose(basis[5, 16], data["cell5"], atol=1e-14)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", grad_m_basis)
    def test_grad_m_basis(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        bcs = mesh.multi_index_matrix(p=4, etype=3)/4
        gmbasis = space.grad_m_basis(bcs, 1)

        np.testing.assert_allclose(gmbasis[0,3,:,0], data["cell0"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[1,22,:,1], data["cell1"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[2,13,:,2],  data["cell2"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[3,33,:,0], data["cell3"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[4,28,:,1], data["cell4"], atol=1e-14)
        np.testing.assert_allclose(gmbasis[5,7,:,2],  data["cell5"], atol=1e-12)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("data", interpolation)
    def test_interpolation(self, data, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],2,2,2)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)

        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        u_sp = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
        np.set_printoptions(threshold=np.inf)
        flist = get_flist(u_sp)
        #torch.set_printoptions(precision=16)
        #torch.set_printoptions(threshold=torch.inf)
        interpolation = space.interpolation(flist)
        np.testing.assert_allclose(interpolation, data["mesh2"], atol=1e-12)
        #a = np.where(np.abs(interpolation-data[1]["mesh2"])>140)
        #print(a)
        
    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_source_vector(self, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 9, 1, isCornerNode)
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        f = x**5*y**4+z**9
        get_flist = get_flist(f)

        @cartesian
        def f(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            return x**5*y**4+z**9
        from fealpy.fem import BilinearForm
        from fealpy.fem import LinearForm, ScalarSourceIntegrator, ScalarMassIntegrator
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(f, q=14))
        F = lform.assembly()

        bform = BilinearForm(space)                                                 
        bform.add_integrator(ScalarMassIntegrator(coef=1,q=14))                     
        M = bform.assembly() #(582, 582) 

        fi = space.interpolation(get_flist)
        x = fi[:]
        aa = M@x
        print(bm.abs(aa-F).max())
        np.testing.assert_allclose(aa, F, atol=1e-14,rtol=0)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    def test_matrix(self, backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        f = x**7*y**2+z**2
        get_flist = get_flist(f)

        from fealpy.fem import BilinearForm
        from fealpy.fem import LinearForm
        from fealpy.fem.scalar_mlaplace_source_integrator import ScalarMLaplaceSourceIntegrator
        from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator
        lform = LinearForm(space)
        lform.add_integrator(ScalarMLaplaceSourceIntegrator(2, get_flist[2], q=14))
        gF = lform.assembly()

        bform = BilinearForm(space)                                                 
        bform.add_integrator(MthLaplaceIntegrator(m=2, coef=1,q=14))                     
        A = bform.assembly()  

        fi = space.interpolation(get_flist)
        x = fi[:]
        aa = A@x
        print(bm.abs(aa-gF).max())
        #np.testing.assert_allclose(aa, gF, atol=1e-14,rtol=0)


    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    #@pytest.mark.parametrize("data", interpolation)
    def test_boundary_interpolation(self,  backend):
        bm.set_backend(backend)
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1],1,1,1)
        node = mesh.entity('node')
        isCornerNode = np.zeros(len(node), dtype=np.bool)
        for n in np.array([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float64):
            isCornerNode = isCornerNode | (np.linalg.norm(node - n[None, :], axis=1) < 1e-10)
        space = CmConformingFESpace3d(mesh, 11, 1, isCornerNode)
        uh = space.function()
        from fealpy.pde.biharmonic_triharmonic_3d import get_flist
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        f = x**7*y**2+z**2
        gD = get_flist(f)

        boundary_interpolation = space.boundary_interpolate(gD, uh)







if __name__=="__main__":

    t = TestCmfespace3d()
    # print(1)
    #t.test_get_dof_index(get_dof_index[0], "numpy")
    # t.test_get_dof_index(get_dof_index[0], "pytorch")
    # t.test_number_of_internal_dofs(number_of_global_dofs, "pytorch")
    #t.test_cell_to_dofs(nefc_to_internal_dof[0], "numpy")
    #t.test_coefficient_matrix(cell2dof, "pytorch")
    #t.test_coefficient_matrix(coefficient_matrix, "numpy")
    #t.test_basis(coefficient_matrix, "numpy")
    #t.test_interpolation(interpolation, "numpy")
    #t.test_interpolation(interpolation, "pytorch")
    #t.test_source_vector("pytorch")
    #t.test_matrix("pytorch")
    t.test_boundary_interpolation("numpy")
