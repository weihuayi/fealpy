import numpy as np
from scipy.fftpack import dct, idct

class UniformTimeLine():
    def __init__(self, T0, T1, NT):
        """
        Parameter
        ---------
        T0: the initial time
        T1: the end time
        NT: the number of time segments
        """
        self.T0 = T0
        self.T1 = T1
        self.NL = NT + 1 # the number of time levels
        self.dt = (self.T1 - self.T0)/NT
        self.current = 0

    def uniform_refine(self, n=1):
        for i in range(n):
            self.NL = 2*(self.NL - 1) + 1
            self.dt = (self.T1 - self.T0)/(self.NL - 1)
        self.current = 0

    def number_of_time_levels(self):
        return self.NL

    def all_time_levels(self):
        return np.linspace(self.T0, self.T1, num=self.NL)

    def current_time_level_index(self):
        return self.current

    def current_time_level(self):
        return self.T0 + self.current*self.dt

    def next_time_level(self):
        return self.T0 + (self.current + 1)*self.dt

    def current_time_step_length(self):
        return self.dt

    def stop(self):
        return self.current >= self.NL - 1

    def advance(self):
        self.current += 1

    def reset(self):
        self.current = 0

    def time_integration(self, data, dmodel, solver):
        timeline = self
        timeline.reset()
        while not self.stop():
            A = dmodel.get_current_left_matrix(data, timeline)
            b = dmodel.get_current_right_vector(data, timeline)
            dmodel.solve(data, A, b, solver, timeline)
            timeline.current += 1
        timeline.reset()

class ChebyshevTimeLine():
    def __init__(self, T0, T1, NT):
        """
        Parameter
        ---------
        T0: the initial time
        T1: the end time
        NT: the number of time segments
        """
        self.T0 = T0
        self.T1 = T1
        self.NL = NT + 1
        self.theta = np.arange(self.NL)*np.pi/NT
        self.time = 0.5*(T0 + T1) - 0.5*(T1 - T0)*np.cos(self.theta)
        self.dt = self.time[1:] - self.time[0:-1]
        self.current = 0

    def uniform_refine(self):
        self.NL = 2*(self.NL - 1) + 1
        NT = self.NL - 1
        self.theta = np.arange(self.NL)*np.pi/NT
        self.time = 0.5*(self.T0 + self.T1) - 0.5*(self.T1 - self.T0)*np.cos(self.theta)
        self.dt = self.time[1:] - self.time[0:-1]
        self.current = 0

    def number_of_time_levels(self):
        return self.NL

    def all_time_levels(self):
        return self.time

    def current_time_level_index(self):
        return self.current

    def current_time_level(self):
        return self.time[self.current]

    def next_time_level(self):
        return self.time[self.current+1]

    def current_time_step_length(self):
        return self.dt[self.current]

    def stop(self):
        return self.current >= self.NL - 1

    def advance(self):
        self.current += 1

    def reset(self):
        self.current = 0

    def diff(self, F):
        d = np.zeros(F.shape, dtype=np.float)
        d[..., 1:] = (F[..., 1:] - F[..., 0:-1])/self.dt
        return d

    def dct_time_integral(self, q, return_all=True):
        """
        q is integrand
        """
        N = self.NL - 1
        theta = self.theta
        a = dct(q, type=1)/N
        if return_all:
            A = np.zeros(a.shape, dtype=np.float)
            A[..., 0] = (
                    a[..., 0] + 0.5*a[..., 1] +
                    np.sum(2*a[..., 2:N]/(1 - np.arange(2, N)**2), axis=-1) +
                    a[..., -1]/(1 - N**2)
                    )
            A[..., 1:N-1] = 0.5*(a[..., 2:N] - a[..., 0:N-2])/np.arange(1, N-1)
            A[..., N-1] = 0.5/(N-1)*(0.5*a[..., N] - a[..., N-2])
            A[..., N] = -0.5/N*a[..., N-1]
            intq = idct(A, type=1)/2 - 0.25*a[..., [-1]]/(N+1)*theta
        else:
            intq = a[..., 0] + np.sum(2*a[..., 2:N:2]/(1 - np.arange(2, N,
                2)**2), axis=-1) + a[..., -1]/(1 - N**2)

        intq *= 0.5*(self.time[-1] - self.time[0])
        return intq

    def time_integration(self, data, dmodel, nupdate=1, F = None):
        """
        F is the cross matrix, sometimes without F
        """
        if F is None:
            F = np.zeros(dmodel.A.shape, dtype=np.float)
            #F = np.zeros(self.NL, dtype=np.float)
        else:
            F = F

        timeline = self
        timeline.reset()
        while not timeline.stop():
            """
            get a initial solution by CN
            """
            dt = timeline.current_time_step_length()
            A = dmodel.get_current_left_matrix(dt, F)
            b = dmodel.get_current_right_vector(data[..., timeline.current], dt, F)
            A, b = dmodel.apply_boundary_condition(A, b)
            data[..., timeline.current+1] = dmodel.solve(A, b)
            timeline.current += 1
        timeline.reset()
        for i in range(nupdate):
            r = dmodel.residual_integration(data, timeline, F)
            if type(data) is not list:
                data = [data, r]
            else:
                data += [r]
            data += [timeline.diff(r)]
            init_error = np.zeros(data[0].shape, dtype=np.float)
            data += [init_error]
            """
            spectral deferred correction
            """
            while not self.stop():
                dt = timeline.current_time_step_length()
                A = dmodel.get_current_left_matrix(dt, F)
                b =dmodel.get_error_right_vector(data[-1][...,timeline.current],
                        dt, data[2][...,timeline.current+1], F)
                A, b = dmodel.apply_boundary_condition(A, b)
                data[-1][...,timeline.current+1] = dmodel.solve(A, b)
                self.current += 1
            self.reset()
            data[0] += data[-1]
            data = data[0]










