import numpy as np

from fealpy.quadrature  import TriangleQuadrature
from fealpy.functionspace.surface_lagrange_fem_space import SurfaceLagrangeFiniteElementSpace
from fealpy.femmodel import doperator 
from scipy.sparse.linalg import spsolve

class SSCFTParameter():
    def __init__(self):
        self.Nspecies = 2 
        self.Nblend   = 1
        self.Nblock   = 2 
        self.Ndeg     = 100  
        self.fA       = 0.8 
        self.chiAB    = 0.3 
        self.dim = 2
        self.dtMax = 0.1
        self.tol = 1.0e-6
        self.tolR = 1.0e-3
        self.maxit = 5000
        self.showstep = 200
        self.pdemethod = 'CN'
        self.integrator = TriangleQuadrature(3)
        self.fieldType = 'fieldmu'


class TimeLine():
    def __init__(self, interval, dt):
        self.T0 = interval[0]
        self.T1 = interval[1]
        self.Nt = int(np.ceil((self.T1 - self.T0)/dt))
        self.dt = (self.T1 - self.T0)/self.Nt
        print(self.dt)
        self.current = 0

    def get_index_interval(self):
        return np.arange(0, self.Nt+1) 

    def get_number_of_time_steps(self):
        return self.Nt+1

    def get_current_time_step(self):
        return self.current

    def get_time_step_length(self):
        return self.dt

    def stop(self):
        return self.current >= self.Nt 

    def step(self):
        self.current += 1

    def reset(self):
        self.current = 0
        
class PDESolver():
    def __init__(self, femspace, integrator, area,  method='FM'):

        self.method = method
        self.femspace = femspace 
        self.integrator = integrator 
        self.area = area

        self.M = doperator.mass_matrix(self.femspace, self.integrator, self.area)
        self.A = doperator.stiff_matrix(self.femspace, self.integrator, self.area)

    def get_current_linear_system(self, u0, dt):
        M = self.M
        print(M,M.shape)
        S = self.A
        print(S,S.shape)
        F = self.F
        print(F,F.shape)
        if self.method is 'FM':
            b = -dt*(S + F)@u0 + M@u0
            print(b)
            A = M   
            print(A)
            return A, b
        if self.method is 'BM':
            b = M@u0
            A = M + dt*(S + F)
            return A, b
        if self.method is 'CN':
            b = -0.5*dt*(S + F)@u0 + M@u0
            A = M + 0.5*dt*(S + F)
            return A, b

    def run(self, timeline, uh, F):
        self.F = F
        while ~timeline.stop(): 
            current = timeline.get_current_time_step()
            dt = timeline.get_time_step_length()
            A, b = self.get_current_linear_system(uh[current], dt)
            self.uh[:, current] = spsolve(A, b)
            timeline.step()
        timeline.reset()
    

class SSCFTFEMModel():
    def __init__(self, surface, mesh, option,fields, p=1, p0=1):
        self.femspace = SurfaceLagrangeFiniteElementSpace(mesh, surface, p=p, p0=p0) 
        self.mesh = self.femspace.mesh
        self.area = mesh.area()
        self.totalArea = np.sum(self.area)
        self.surface = surface
        self.option = option
        self.fields = fields

        self.timeline0 = TimeLine([0, option.fA], option.dtMax)
        self.timeline1 = TimeLine([option.fA, 1], option.dtMax)

        N = self.timeline0.Nt + self.timeline1.Nt + 1
        self.q0 = self.femspace.function(dim=N) 
        self.q1 = self.femspace.function(dim=N) 

        self.rho   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.w   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.mu   = [self.femspace.function() for i in range(option.Nspecies)] 
        self.grad = self.femspace.function(dim=option.Nspecies)
        self.sQ    = np.zeros((option.Nspecies-1, option.Nblend))

        self.solver = PDESolver(self.femspace, option.integrator, self.area, option.pdemethod)

        self.initialize()

    def initialize(self):
        option = self.option
        fields = self.fields
        chiN = option.chiAB * option.Ndeg
         
        if option.fieldType is 'fieldmu':
            self.mu[0][:]   = fields[:, 0]
           # print('mu0', fields[:, 0])
            self.mu[1][:]   = fields[:, 1]
           # print('mu1',fields[:, 1])
            self.w[0][:] = fields[:, 0] - fields[:, 1]
            self.w[1][:] = fields[:, 0] + fields[:, 1]
        if option.fieldType is 'fieldw':
            self.w[0][:]   = fields[:, 0]
            self.w[1][:]   = fields[:, 1]
            self.mu[0][:] = 0.5*(fields[:, 0] + fields[:, 1])
            self.mu[1][:] = 0.5*(fields[:, 1] - fields[:, 0])

        self.rho[0][:] = 0.5 + self.mu[1]/chiN
        self.rho[1][:] = 1.0 - self.rho[0]
        print(self.rho)

    def update_propagator(self):
        option = self.option
        n0 = self.timeline0.get_number_of_time_steps()
        print(n0)
        n1 = self.timeline1.get_number_of_time_steps()

        F0 = doperator.mass_matrix(
                self.femspace, 
                option.integrator, 
                self.area, 
                cfun=self.w[0].value)

        F1 = doperator.mass_matrix(
                self.femspace, 
                option.integrator, 
                self.area, 
                cfun=self.w[1].value)

        self.q0[:, 0] = 1.0
        print( self.solver.run(self.timeline0, self.q0[:, 0:n0], F0))
        self.solver.run(self.timeline1, self.q0[:, n0-1:], F1)
        self.q1[:, 0] = 1.0
        self.solver.run(self.timeline1, self.q1[:, 0:n1], F1)
        self.solver.run(self.timeline0, self.q1[:, n1-1:], F0)

    def integral_time(self, q, dt):
        f = -0.625*(q[:, 0] + q[:, -1]) + 1/6*(q[:, 1] + q[:, -2]) + 1/24*(q[:,
            2] + q[:, -3])
        f = f + np.sum(q, axis=1)
        f = dt*f
        return f
    
    def integral_space(self, q):
        mesh = self.femspace.mesh
        qf = self.option.integrator 
        bcs, ws = qf.quadpts, qf.weights
        qq = q.value(bcs)
        b = np.einsum('i, ij->j', ws, qq)
        b *= self.area
        Q = np.sum(b)
        return Q

    def integral_space2(self, q):
        mesh = self.femspace.mesh
        qf = self.option.integrator 
        bcs, ws = qf.quadpts, qf.weights
        qq = q.value(bcs)**2
        b = np.einsum('i, ij->j', ws, qq)
        b *= self.area
        Q = np.sum(b)
        return Q

    def update_density(self, q):
        n0 = self.timeline0.get_number_of_time_steps()
        self.rho[0][:] = integral_time(q[:, 0:n0], self.timeline0.dt)
        self.rho[1][:] = integral_time(q[:, n0-1:], self.timeline1.dt)

    def update_singleQ(self, integrand):
        f = self.integral_space(integrand)/self.totalArea 
        return f

    def update_hamilton(self):
        chiN = self.chiAB * self.Ndeg
        mu1_int = self.integral_space(self.mu[0])
        mu2_int = self.integral_space2(self.mu[1])
        H = -mu1_int + mu2_int/chiN
        return H

    def updateField(self):

        option = self.option
        chiN = option.chiAB * option.Ndeg
        
        lambd = np.array([2,2])

        self.grad[:, 0] = self.rho[0]  + self.rho[1] - 1.0
        self.grad[:, 1] = 2.0*self.mu[1]/chiN - self.rho[0] + self.rho[1]

        err = np.max(np.abs(self.grad), axis=0)

        self.mu[0] += lambd[0]*self.grad[:, 0]
        self.mu[1] -= lambd[1]*self.grad[:, 1]


        self.w[0] = self.mu[0] - self.mu[1] 
        self.w[1] = self.mu[1] + self.mu[1]

        return err 

    def find_saddle_point(self):
        
        res = np.inf
        Hold = np.inf
        ediff = np.inf
        iteration = 0

        error = np.arange(self.grad.shape[0])
        option = self.option

        while (res > option.tol) and (iteration < option.maxit):
            
            self.update_propagator() 

            qq = self.q0*self.q1[:, -1::-1] 

            self.sQ[0,0] = self.update_singleQ(self, self.q0.index(-1))

            self.update_density(qq)

            error = self.update_field()
        
            H = self.update_hamilton()

            H = H/self.totalArea - np.log(self.sQ(0,0))

            res = error.max()                      

            ediff = H - Hold
            Hold = H

    def get_partial_F(self, radius):
 
        self.update_propagator()

        qq = self.q0 * self.q1[:, -1::-1] 
        self.sQ[0,0] = self.updateQ(self, qq)
        gradF = -np.sum(sQ.reshape(-1))

        return partialF
