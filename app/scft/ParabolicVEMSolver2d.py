import numpy as np
from fealpy.solver.petsc_solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg

class ParabolicVEMSolver2d():
    """
    这是一个求解如下抛物方程:
    q_t = \Delta q - w*q
    q[0] = 1
    """
    def __init__(self, A, M, F=None, nupdate=0, method ='CN'):
        self.method = method
        self.solver = PETScSolver()

        self.A = A
        self.M = M
        self.F = F

        self.nupdate = nupdate

    def get_current_left_matrix(self, dt):
        M = self.M
        S = self.A
        F = self.F
        return M + 0.5*dt*(S + F)

    def get_current_right_vector(self, u0, dt):
        M = self.M
        S = self.A
        F = self.F
        return -0.5*dt*(S@u0 + F@u0) + M@u0

    def get_error_right_vector(self, data, dt, diff):
        b = self.get_current_right_vector(data, dt) + dt*self.M@diff
        return b


    def apply_boundary_condition(self, A, b):
        return A,b

    def residual_integration(self, data, timeline):
        F = self.F
        q = -self.A@data - F@data
        r = self.M[0,...]@data[...,0] + timeline.dct_time_integral(q) - self.M@data
        return r

    def solve(self, data, timeline):
        current = timeline.current
        dt = timeline.current_time_step_length()
        A = self.get_current_left_matrix(dt)
        b = self.get_current_right_vector(data[:,current], dt)
        A, b = self.apply_boundary_condition(A, b)
        data[:,current+1]=spsolve(A,b)
        #self.solver.solve(A, b, data[:,current+1])

    def correct_solve(self, data, timeline):
        current = timeline.current
        dt = timeline.current_time_step_length()
        A = self.get_current_left_matrix(dt)
        b = self.get_error_right_vector(data[-1], dt, data[2][:,current+1])
        A, b = self.apply_boundary_condition(A, b)
        data[-1]=spsolve(A,b)
        #self.solver.solve(A, b, data[-1])

    def output(self, data, nameflag, queue=None, stop=False):
        if queue is not None:
            if not stop:
                queue.put({'u'+nameflag: data})
            else:
                queue.put(-1)
