import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..quadrature  import TriangleQuadrature
from ..functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace

from fealpy.timeintegratoralg.TimeIntegratorAlgorithm import TimeIntegratorAlgorithm
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from ..solver import solve
from ..boundarycondition import DirichletBC

class SurfaceHeatFEMModel(TimeIntegratorAlgorithm):
    def __init__(self, mesh, surface, model, initTime, stopTime, N,
            method='FM', integrator=None, p=1,p0=None):
        super(SurfaceHeatFEMModel, self).__init__(initTime, stopTime)

        """
        surface parabolic equation
        """
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=p, p0=p0) 
        self.mesh = self.V.mesh
        self.surface = surface
        self.model = model
        self.N = N
        self.dt = (stopTime - initTime)/N
        self.method = method
        self.uh = self.V.function() 
        self.uh[:] = model.init_value(self.V.interpolation_points())

        self.solution = [self.uh]
        self.maxError = 0.0
       # self.stiffMatrix = get_left_matrix()
       # self.massMatrix = get_mass_matrix()

        if integrator is None:
            self.integrator = TriangleQuadrature(p+1)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator
        self.area = self.V.mesh.area(integrator)
    
    def reinit(self, mesh, p=None):
        if p is None:
            p = self.V.p
        self.V = SurfaceLagrangeFiniteElementSpace(mesh, self.surface, p)
        self.area = self.V.mesh.area(self.integrator)
    

    def get_left_matrix(self):
        """
        compute stiffMatrix
        """
        V = self.V
        mesh = self.mesh
        gdof = V.number_of_global_dof()
        lodf = V.number_of_local_dofs()
        cell2dof = V.dof.cell2dof
        area = self.area

        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        gphi = V.grad_basis(bcs)
        A = np.einsum('i, ijkm, ijpm->jkp', ws, gphi, gphi)
        A *= area.reshape(-1, 1, 1)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def get_mass_matrix(self):
        """
        comupte massMatrix
        """
        V = self.V
        mesh = self.mesh
        gdof = V.number_of_global_dof()
        ldof = V.number_of_local_dofs()
        cell2dof = V.dof.cell2dof
        area = self.area
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        phi = V.basis(bcs)
        M = np.einsum('i, ijkm, ijpm->jkp', ws, phi, phi)
        M *= area.reshape(-1, 1, 1)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2) 
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def get_right_vector(self):
        V = self.V
        mseh = self.mesh
        model = self.model
        model = lambda p: model.source(p, t)
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)
        fval = model.source(pp)
        phi = V.basis(bcs)
        bb = np.einsum('i, ij, ik->kj', ws, phi, fval)
        bb *= self.area.reshape(-1, 1)
        gdof = V.number_of_global_dofs()
        b = np.bincount(V.dof.cell2dof.flat, weights=bb.flat, minlength=gdof)
        b -= np.mean(b)
        return b
    
    def get_setp_length(self):
        return self.dt

    def get_current_linear_system(self):
        V = self.V
        model = self.model
        t = self.currentTime

        self.stiffMatrix = get_left_matrix()
        self.massMatrix = get_mass_matrix()
        
        S = self.stiffMatrix
        M = self.massMatrix
        K = model.diffusion_coefficient()
        g0 = lambda p: 0 
        bc = DirichletBC(self.V, g0, self.is_boundary_dof)
        F = get_right_vector() 
        dt = self.dt
        if self.method is 'FM':
            b = dt*(F - K*S@self.solution[-1]) + M@self.solution[-1]
            A, b = bc.apply(M, b)
            return A, b
        if self.method is 'BM':
            b = dt*F + M@self.solution[-1]
            A = M + dt*K*S
            A, b = bc.apply(A, b) 
            return A, b
        if self.method is 'CN':
            b = dt*F + (M-0.5*dt*K*S)@self.solution[-1]
            A = M + 0.5*dt*K*S
            A, b = bc.apply(A, b)
            return A, b

    def accept_solution(self, currentSolution):
        self.solution.append(currentSolution)

        V = self.V
        model = self.model
        t = self.currentTime
        u = model.solution(V.interpolation_point(),t)
        e = np.maxError = max(e, self.maxError)
    
    def solve(self, A, b):
        self.uh = self.V.functionspace()
        self.uh[:] = spsolve(A, b)
        return uh

    def is_boundary_dof(self,p):
        isBdDof = np.zeros(p.shape[0],dtype=np.bool)
        isBdDof[0] = True
        return isBdDof

