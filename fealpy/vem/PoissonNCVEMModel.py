import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace import NonConformingVirtualElementSpace2d
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..quadrature import GaussLegendreQuadrature


class PoissonNCVEMModel():
    def __init__(self, pde, mesh, p=1, q=None):
        """
        Initialize a Poisson non-conforming virtual element model.

        Parameters
        ----------
        self : PoissonVEMModel object
        pde :  PDE Model object
        mesh : PolygonMesh object
        p : int

        See Also
        --------

        Notes
        -----
        """
        self.space = NonConformingVirtualElementSpace2d(mesh, p, q)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.area = self.mesh.entity_measure("cell")

        self.uI = self.space.interpolation(pde.solution)

    def reinit(self, mesh, p, q=None):
        self.space =NonConformingVirtualElementSpace2d(mesh, p, q)
        self.mesh = self.space.mesh
        self.uh = self.space.function()

        self.uI = self.space.interpolation(self.pde.solution)

    def get_left_matrix(self):
        return self.space.stiff_matrix()

    def get_right_vector(self):
        f = self.pde.source
        return self.space.source_vector(f)

    def solve(self):
        uh = self.uh
        bc = DirichletBC(self.space, self.pde.dirichlet)
        self.A, b = solve(self, uh, dirichlet=bc, solver='direct')

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self):
        u = self.pde.solution
        S = self.space.project_to_smspace(self.uh)
        uh = S.value
        return self.space.integralalg.L2_error(u, uh)

    def H1_semi_error(self):
        gu = self.pde.gradient
        S = self.space.project_to_smspace(self.uh)
        guh = S.grad_value
        e = self.space.integralalg.L2_error(gu, guh)
        return e

    def stability_term(self):
        space = self.space
        area = self.area

        G = space.G
        PI0 = space.PI0
        PI1 = space.PI1
        D = space.D

        cell2dof, cell2dofLocation = space.dof.cell2dof, space.dof.cell2dofLocation
        uh = self.uh[cell2dof]
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        uh = np.hsplit(uh, cell2dofLocation[1:-1])

        def f0(x):
            val = (np.eye(x[1].shape[1]) - x[0]@x[1])@x[2]
            return np.sum(val*val)
        psi0 = sum(map(f0, zip(DD, PI1, uh)))

        def f1(x):
            val = (np.eye(x[1].shape[1]) - x[0]@x[1])@x[2]
            return x[3]*np.sum(val*val)
        psi1 = sum(map(f1, zip(DD, PI0, uh, area)))

        return psi0, psi1

