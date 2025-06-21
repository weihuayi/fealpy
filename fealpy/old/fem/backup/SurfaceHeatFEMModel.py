import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from scipy.sparse.linalg import cg, inv, dsolve, spsolve
from ..femmodel import doperator 


class TimeModel():
    def __init__(self, w, q0):
        self.w = w
        self.q0 = q0
        self.timeline 
        self.current = 0

    def get_number_of_time_steps(self):
        return len(self.timeline)

    def get_current_time_step(self):
        return self.current

    def get_time_step_length(self):
        return self.timeline[self.current+1] - self.timeline[self.current] 

    def stop(self):
        return self.current >= len(self.timeline) - 1 

    def step(self):
        self.current += 1



class SurfaceHeatFEMModel():
    def __init__(self, tmodel, V, method='FM', integrator):
        """
        surface parabolic equation
        """

        self.method = method
        self.tmodel = tmodel
        self.V = V
        self.mesh = self.V.mesh

        self.uh = self.V.function(dim=NT) 
        self.solution = [self.uh]
        self.integrator = integrator 
        self.area = self.V.mesh.area(integrator)

        self.M = doperator.mass_matrix(self.V, self.integrator, self.area)
        self.A = doperator.stiff_matrix(self.V, self.integrator, self.area)
        self.F = doperator.mass_matrix(self.V, self.integrator, self.area,
                cfun=tmodel.w.value)


    def get_current_linear_system(self):
        
        M = self.M
        S = self.A
        F = self.F
        dt = self.tmodel.get_time_step_length()
        # there have diffconDiffusion coefficient and Radius of gyration
        if self.method is 'FM':
            b = -dt*(S + F)@self.solution[-1] + M@self.solution[-1]
            A = M                                                         
            return A, b
        if self.method is 'BM':
            b = M@self.solution[-1]
            A = M + dt*(S + F)
            return A, b
        if self.method is 'CN':
            b = -0.5*dt(S + F)@self.solution[-1] + M@self.solution[-1]
            A = M + 0.5*dt(S + F)
            return A, b

    def run(self):
        while ~self.tmodel.stop(): 
            current = self.tmodel.get_current_time_step()
            A, b = self.get_current_linear_system()
            self.uh[:, self.tmodel.current] = spsolve(A, b)
            self.tmodel.step()
    
    def solve(self, A, b):
        uh  = spsolve(A, b)
        return uh 


