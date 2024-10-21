
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace  
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace  
from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..boundarycondition import DirichletBC

class PoissonRecoveryFEMModel:
    def __init__(self, mesh, model, integrator=None, rtype='simple'):
        self.V = LagrangeFiniteElementSpace(mesh, p=1) 
        self.V2 = VectorLagrangeFiniteElementSpace(mesh, p=1, vectordim=2)

        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution) 
        self.rgh = self.V2.function()

        self.model = model
        if integrator is None:
            self.integrator = TriangleQuadrature(3)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 
        self.rtype = rtype 

        self.area = mesh.area()
        self.form = Form()


    def reinit(self, mesh):
        self.V = LagrangeFiniteElementSpace(mesh, p=1) 
        self.V2 = VectorLagrangeFiniteElementSpace(mesh, p=1)
        self.uh = self.V.function()
        self.uI = self.V.interpolation(self.model.solution)
        self.rgh = self.V2.function()
        self.area = self.V.mesh.area()

    def recover_grad(self):
        uh = self.uh
        self.rgh[:, 0] = self.A@uh
        self.rgh[:, 1] = self.B@uh

    def get_left_matrix(self):
        M = self.form.mass_matrix(self)
        A, B = self.form.grad_recovery_matrix(self)
        D = A.transpose()@M@A + B.transpose()@M@B
        self.A = A
        self.B = B
        return D

    def get_right_vector(self):
        return self.form.source_vector(self)

    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')
