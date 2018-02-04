
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..functionspace.vem_space import VirtualElementSpace2d 
from ..solver import active_set_solver 
from ..boundarycondition import DirichletBC
from ..vemmodel import doperator, PolygonMeshIntegralAlg

class ObstacleVEMModel2d():
    def __init__(self, model, mesh, p=1, integrator=None):
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
        self.vemspace =VirtualElementSpace2d(mesh, p) 
        self.mesh = self.vemspace.mesh
        self.model = model  

        self.integrator = integrator
        self.area = self.vemspace.smspace.area 

        self.uh = self.vemspace.function() 

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.vemspace.smspace.barycenter)

        self.gI = self.vemspace.interpolation(model.obstacle, self.integralalg.integral)
        self.uI = self.vemspace.interpolation(model.solution, self.integralalg.integral)

        self.H = doperator.matrix_H(self.vemspace)

        self.D = doperator.matrix_D(self.vemspace, self.H)
        self.B = doperator.matrix_B(self.vemspace)
        self.G = doperator.matrix_G(self.vemspace, self.B, self.D)
        self.C = doperator.matrix_C(self.vemspace, self.B, self.D, self.H, self.area)

        self.PI0 = doperator.matrix_PI_0(self.vemspace, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.vemspace, self.G, self.B)


    def reinit(self, mesh, p=None):
        if p is None:
            p = self.vemspace.p
        self.vemspace = VirtualElementSpace2d(mesh, p) 
        self.mesh = self.vemspace.mesh
        self.uh = self.vemspace.function() 
        self.area = self.vemspace.smspace.area
        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator, 
                self.mesh, 
                area=self.area, 
                barycenter=self.vemspace.smspace.barycenter)
        self.gI = self.vemspace.interpolation(self.model.obstacle, self.integralalg.integral)
        self.uI = self.vemspace.interpolation(self.model.solution, self.integralalg.integral)

        self.H = doperator.matrix_H(self.vemspace)
        self.D = doperator.matrix_D(self.vemspace, self.H)
        self.B = doperator.matrix_B(self.vemspace)
        self.C = doperator.matrix_C(self.vemspace, self.B, self.D, self.H, self.area)

        self.G = doperator.matrix_G(self.vemspace, self.B, self.D)

        self.PI0 = doperator.matrix_PI_0(self.vemspace, self.H, self.C)
        self.PI1 = doperator.matrix_PI_1(self.vemspace, self.G, self.B)

    def project_to_smspace(self, uh=None):
        p = self.vemspace.p
        cell2dof, cell2dofLocation = self.vemspace.dof.cell2dof, self.vemspace.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@self.uh[x[1]]
        S = self.vemspace.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def get_left_matrix(self):
        vemspace = self.vemspace
        area = self.area
        return doperator.stiff_matrix(vemspace, area, vem=self)

    def get_right_vector(self):
        f = self.model.source
        integral = self.integralalg.integral 
        return doperator.source_vector(
                integral,
                f, 
                self.vemspace,
                self.PI0)

    def solve(self):
        uh = self.uh
        gI = self.gI
        bc = DirichletBC(self.vemspace, self.model.dirichlet)
        active_set_solver(self, uh, gI, dirichlet=bc, solver='direct')
        self.S = self.project_to_smspace(uh)

    def l2_error(self):
        e = self.uh - self.uI
        return np.sqrt(np.mean(e**2))

    def uIuh_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)

    def L2_error(self):
        u = self.model.solution
        uh = self.S.value
        return self.integralalg.L2_error(u, uh)

    def H1_semi_error(self):
        gu = self.model.gradient
        guh = self.S.grad_value
        return self.integralalg.L2_error(gu, guh)

