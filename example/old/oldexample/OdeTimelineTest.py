import sys
import numpy as np

from scipy.sparse.linalg import spsolve

from fealpy.timeintegratoralg.timeline import UniformTimeLine, ChebyshevTimeLine

class OdeSolver():
    """
    This is an example for timeline
    ode: u_t = -u
    u(0) = 1
    exact solution: u =e^(-t)
    """
    def __init__(self):
        pass

    def u_exact(self, t):
        return np.exp(-t)

    def get_current_left_matrix(self, dt):
        return 1 + 0.5*dt

    def get_current_right_vector(self, u0, dt):
        return u0-0.5*dt*u0

    def get_error_right_vector(self, data, dt, diff):
        b = self.get_current_right_vector(data, dt) + dt*diff
        return b

    def apply_boundary_condition(self, A, b):
        return A,b

    def residual_integration(self, data, timeline):
        q = -data
        r = data[0] + timeline.dct_time_integral(q) - data
        return r

    def solve(self, data, timeline):
        current = timeline.current
        dt = timeline.current_time_step_length()
        A = self.get_current_left_matrix(dt)
        b = self.get_current_right_vector(data[current], dt)
        A,b = self.apply_boundary_condition(A,b)
        data[current+1] = b/A

    def correct_solve(self, data, timeline):
        current = timeline.current
        dt = timeline.current_time_step_length()
        A = self.get_current_left_matrix(dt)
        b = self.get_error_right_vector(data[-1][current], dt, data[2][current+1])
        A, b = self.apply_boundary_condition(A, b)
        data[-1][current+1]=b/A
        data[0][current+1] += data[-1][current+1]

    def output(self, data, nameflag, queue=None, stop=False):
        if queue is not None:
            if not stop:
                queue.put({'u'+nameflag: data})
            else:
                queue.put(-1)


T=5
maxit = 5
nupdate = int(sys.argv[1])
eold = np.inf
for i in range(maxit):
    T=2*T
    timeline = ChebyshevTimeLine(0,1,T)
    #timeline = UniformTimeLine(0,1,T)
    smodel = OdeSolver()
    t = timeline.time
    ue = smodel.u_exact(t)
    u_init = np.ones(T+1)
    timeline.time_integration(u_init, smodel, nupdate=nupdate)
    #timeline.time_integration(u_init, smodel)
    e = np.max(abs(u_init - ue))
    print('uinit:',u_init)
    print('ue',ue)
    order = np.log2(eold/e)
    eold = e
    print(order)


