import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import FiniteElementFunction
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace
from ..quadrature  import TriangleQuadrature
from ..solver import solve
from ..boundarycondition import DirichletBC
from ..femmodel import form 

class PoissonFEMModel(object):
    def __init__(self, mesh,  model, integrator=None, p=1):
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.model = model
        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution)
        self.area = mesh.area()

        if integrator is None:
            self.integrator = TriangleQuadrature(3)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 

    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = LagrangeFiniteElementSpace(mesh, p) 
        self.uh = self.V.function()
        self.uI = self.V.interpolation(self.model.solution)
        self.area = mesh.area()

    def recover_estimate(self):
        if self.V.p > 1:
            raise ValueError('This method only work for p=1!')

        V = self.V
        mesh = V.mesh

        p2c = mesh.ds.point_to_cell()
        inva = 1/mesh.area()
        asum = p2c@inva

        bc = np.array([1/3]*3, dtype=np.float)
        guh = self.uh.grad_value(bc)

        VV = VectorLagrangeFiniteElementSpace(mesh, p=1)
        rguh = VV.function()
        rguh[:] = np.asarray(p2c@(guh*inva.reshape(-1, 1)))/asum.reshape(-1, 1)

        qf = self.integrator  
        bcs, ws = qf.quadpts, qf.weights

        val0 = rguh.value(bcs)
        val1 = self.uh.grad_value(bcs)
        l = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, l)
        e *= self.area
        return np.sqrt(e)

    
    def get_left_matrix(self):
        return form.stiff_matrix(self)

    def get_right_vector(self):
        return form.source_vector(self)

    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')


