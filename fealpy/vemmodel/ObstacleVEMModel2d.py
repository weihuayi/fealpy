
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import active_set_solver 
from ..boundarycondition import DirichletBC
from ..vemmodel import doperator 
from ..functionspace import FunctionNorm

class ObstacleVEMModel2d():
    def __init__(self, model, mesh, p=1, integrator=None, quadtree=None):
        """
        Initialize a Poisson virtual element model. 

        Parameters
        ----------
        self : PoissonVEMModel object
        model :  PDE Model object
        mesh : PolygonMesh object
        p : int
        
        See Also
        --------

        Notes
        -----
        """
        self.V =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.V.mesh
        self.quadtree = quadtree
        self.model = model  

        self.area = self.V.smspace.area 
        self.error = FunctionNorm(integrator, self.area)

        self.uh = self.V.function() 
        self.gI = self.V.interpolation(model.obstacle)
        self.uI = self.V.interpolation(model.solution)
        if p == 2:
            NC = self.mesh.number_of_cells()
            self.gI[-NC:] = self.error.integral(model.obstacle, self.quadtree, elemtype=True)/self.area
            self.uI[-NC:] = self.error.integral(model.solution, self.quadtree, elemtype=True)/self.area

            

        self.H = doperator.matrix_H(self.V)

        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)
        self.G = doperator.matrix_G(self.V, self.B, self.D)
        self.C = doperator.matrix_C(self.V, self.B, self.D, self.H, self.area)

        self.PI0 = doperator.matrix_PI_0(self.V, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.V, self.G, self.B)

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.V.mesh
        self.uh = self.V.function() 
        self.gI = self.V.interpolation(self.model.obstacle)
        self.uI = self.V.interpolation(self.model.solution)
        self.area = self.V.smspace.area
        self.error.area = self.area 

        if p == 2:
            NC = self.mesh.number_of_cells()
            self.gI[-NC:] = self.error.integral(self.model.obstacle, self.quadtree, elemtype=True)/self.area
            self.uI[-NC:] = self.error.integral(self.model.solution, self.quadtree, elemtype=True)/self.area

        self.H = doperator.matrix_H(self.V)
        self.D = doperator.matrix_D(self.V, self.H)
        self.B = doperator.matrix_B(self.V)
        self.C = doperator.matrix_C(self.V, self.B, self.D, self.H, self.area)

        self.G = doperator.matrix_G(self.V, self.B, self.D)

        self.PI0 = doperator.matrix_PI_0(self.V, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.V, self.G, self.B)

    def project_to_smspace(self, uh=None):
        p = self.V.p
        cell2dof, cell2dofLocation = self.V.dof.cell2dof, self.V.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@self.uh[x[1]]
        S = self.V.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def get_left_matrix(self):
        V = self.V
        area = self.area
        return doperator.stiff_matrix(V, area, vem=self)

    def get_right_vector(self):
        V = self.V
        area = self.area
        f = self.model.source
        return doperator.source_vector(f, V, area, vem=self)

    def solve(self):
        uh = self.uh
        gI = self.gI
        bc = DirichletBC(self.V, self.model.dirichlet)
        active_set_solver(self, uh, gI, dirichlet=bc, solver='direct')
        self.S = self.project_to_smspace(uh)

    def l2_error(self):
        u = self.model.solution
        uh = self.uh
        return self.error.l2_error(u, uh)

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self):
        u = self.model.solution
        uh = self.S.value
        return self.error.L2_error(u, uh, self.quadtree, barycenter=False)

    def H1_semi_error(self):
        gu = self.model.gradient
        guh = self.S.grad_value
        return self.error.L2_error(gu, guh, self.quadtree, barycenter=False)

