
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
import pyamg

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..boundarycondition import DirichletBC
from ..vem import doperator 
from ..quadrature import GaussLobattoQuadrature, PolygonMeshIntegralAlg


class SFCVEMModel2d():
    def __init__(self, pde, mesh, p=1, q=4):
        """
        Initialize a vem model for the simplified friction problem. 

        Parameters
        ----------
        self: PoissonVEMModel object
        pde:  PDE Model object
        mesh: PolygonMesh object
        p: int
        q: 
        
        See Also
        --------

        Notes
        -----
        """
        self.space = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.space.mesh
        self.pde = pde  

        self.integrator = mesh.integrator(q)
        self.area = self.space.smspace.area 

        self.uh = self.space.function() 
        self.lh = self.space.function()

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.space.smspace.barycenter)

        self.mat = doperator.basic_matrix(self.space, self.area)

        self.errorType = ['$\| u - \Pi^\\nabla u_h\|_0$', '$\|\\nabla u - \\nabla \Pi^\\nabla u_h\|$']

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.space.p
        self.space = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.space.mesh

        self.uh = self.space.function() 
        self.lh = self.space.function()

        self.area = self.space.smspace.area
        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.space.smspace.barycenter)

        self.mat = doperator.basic_matrix(self.space, self.area)

    def project_to_smspace(self, uh=None):
        if uh is None:
            uh = self.uh
        p = self.space.p
        cell2dof, cell2dofLocation = self.space.dof.cell2dof, self.space.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@uh[x[1]]
        S = self.space.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.mat.PI1, cd))))
        return S

    def get_left_matrix(self):
        space = self.space
        area = self.area
        A = doperator.stiff_matrix(space, area, mat=self.mat)
        M = doperator.mass_matrix(space, area, mat=self.mat)
        return A+M

    def get_right_vector(self):
        f = self.pde.source
        integral = self.integralalg.integral 
        return doperator.source_vector(
                integral,
                f, 
                self.space,
                self.mat.PI0)

    def get_lagrangian_multiplier_vector(self, cedge, cedge2dof):
        p = self.space.p
        node = self.mesh.node
        v = node[cedge[:, 1]] - node[cedge[:, 0]] 
        l = np.sqrt(np.sum(v**2, axis=1))
        qf = GaussLobattoQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights 
        lh = self.lh
        bb = np.einsum('i, ji, j->ji', ws, lh[cedge2dof], l)
        
        gdof = self.space.number_of_global_dofs()
        b = np.bincount(cedge2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def solve(self, rho=1, maxit=10000, tol=1e-8):
        uh = self.uh
        lh = self.lh

        space = self.space
        mesh = self.mesh

        edge = mesh.ds.edge
        edge2dof = space.dof.edge_to_dof()
        bc = mesh.entity_barycenter(etype='edge') 
        isContactEdge = self.pde.is_contact(bc)

        edge = edge[isContactEdge]
        edge2dof = edge2dof[isContactEdge]

        gdof = space.number_of_global_dofs()
        isContactDof = np.zeros(gdof, dtype=np.bool)
        isContactDof[edge2dof] = True

        bc = DirichletBC(self.space, self.pde.dirichlet,
                is_dirichlet_dof=self.pde.is_dirichlet)

        A = self.get_left_matrix()
        b = self.get_right_vector() 

        k = 0
        g = self.pde.g

        AD = bc.apply_on_matrix(A)
        ml = pyamg.ruge_stuben_solver(AD)  
        while k < maxit:
            b1 = self.get_lagrangian_multiplier_vector(edge, edge2dof)
            bd = bc.apply_on_vector(b - g*b1, A)
            uh0 = uh.copy()
            lh0 = lh.copy()
            uh[:] = ml.solve(bd, tol=1e-12, accel='cg').reshape(-1)
            lh[isContactDof] = np.clip(lh[isContactDof] + rho*g*uh[isContactDof], -1, 1) 
            e0 = np.sqrt(np.sum((uh - uh0)**2))
            e1 = np.sqrt(np.sum((lh - lh0)**2))
            print('k:', k, 'error:', e0, e1)
            if e0 < tol:
                break
            k += 1

        self.S = self.project_to_smspace(uh)

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self, u):
        uh = self.S.value
        def f(x, cellidx):
            return (u(x, cellidx) - uh(x, cellidx))**2

        e = self.integralalg.integral(f, celltype=True)
        return np.sqrt(e.sum())

    def H1_semi_error(self, gu):
        guh = self.S.grad_value
        def f(x, cellidx):
            return (gu(x, cellidx) - guh(x, cellidx))**2
        e = self.integralalg.integral(f, celltype=True)
        return np.sqrt(e.sum())

