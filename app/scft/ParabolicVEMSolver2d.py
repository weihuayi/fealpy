import numpy as np
from fealpy.solver import MatlabSolver
from scipy.sparse.linalg import spsolve
import pyamg

class ParabolicVEMSolver2d():
    """
    这是一个求解如下抛物方程:
    q_t = \Delta q - w*q
    q[0] = 1
    """
    def __init__(self, A, M, nupdate=0, method ='CN'):
        self.method = method
        self.solver = MatlabSolver

        self.A = A
        self.M = M

        self.nupdate = nupdate

    def get_current_linear_system(self, u0, dt, F):
        M = self.M
        S = self.A
        #F = self.F
        if self.method is 'FM':
                b = -dt*(S + F)@u0 + M@u0
                A = M
                return A, b
        if self.method is 'BM':
                b = M@u0
                A = M + dt*(S + F)
                return A, b
        if self.method is 'CN':
                b = -0.5*dt*(S + F)@u0 + M@u0
                A = M + 0.5*dt*(S + F)
                return A, b

    def get_current_left_matrix(self, dt,F):
        M = self.M
        S = self.A
        #F = self.F
        return M + 0.5*dt*(S + F)

    def get_current_right_vector(self, u0, dt, F):
        M = self.M
        S = self.A
        #F = self.F
        return -0.5*dt*(S@u0 + F@u0) + M@u0

    def get_error_right_vector(self, data, dt, diff,F):
        b = self.get_current_right_vector(data, dt, F) + dt*self.M@diff
        return b


    def apply_boundary_condition(self, A, b):
        return A,b

    def residual_integration(self, data, timeline, F):
        q = -self.A@data - F@data
        r = self.M[0,...]@data[...,0] + timeline.dct_time_integral(q) - self.M@data
        return r

    def solve(self, A, b):
        data = spsolve(A,b).reshape((-1,))
        #data = self.solver.divide(A, b)

    def run(self, timeline, uh, F):
        #self.solver =[]
        while not timeline.stop():
            current = timeline.current
            #dt  = timeline.current_time_step_length()
            dt = timeline.get_current_time_step_length()
            A = self.get_current_left_matrix(dt,F)
            #ml = pyamg.ruge_stuben_solver(A) # 多重网格计算 2 次元有问题
            #self.solver.append(ml)

            b = self.get_current_right_vector(uh[:, current], dt,F)
            #uh[:, current+1] = ml.solve(b, tol=1e-12, accel='cg').reshape((-1,))
            uh[:, current+1] = spsolve(A,b).reshape((-1,))
            #uh[:, current+1] = self.solver.divide(A, b)
            timeline.current += 1
        timeline.reset()

        for i in range(self.nupdate):
            q = -self.A@uh - F@uh
            intq = timeline.time_integral(q)
            r = uh[:, [0]] + spsolve(self.M,intq) - uh
            d = timeline.diff(r)
            delta = np.zeros(uh.shape, dtype=np.float)
            while not timeline.stop():
                current = timeline.current
                dt = timeline.get_current_time_step_length()
                A = self.get_current_left_matrix(dt,F)
                b = self.get_current_right_vector(delta[:, current], dt, F) + dt*self.M@d[:, current+1]
                #delta[:, current+1] = self.solver[current].solve(b, tol=1e-12, accel='cg').reshape((-1,))
                delta[:, current+1] = spsolve(A,b).reshape((-1,))
                #delta[:, current+1] = self.solver.divide(A,b)
                timeline.current += 1
            timeline.reset()
            uh += delta
