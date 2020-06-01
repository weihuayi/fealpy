import numpy as np
from scipy.fftpack import dct, idct


class UniformTimeLine():
    def __init__(self, T0, T1, N):
        self.T0 = T0
        self.T1 = T1
        self.NT = N + 1
        self.dt = (self.T1 - self.T0)/N
        self.current = 0

    def get_number_of_time_steps(self):
        return self.NT

    def get_current_time_step(self):
        return self.current

    def get_current_time(self):
        return self.T0 + self.current*self.dt

    def get_next_current_time(self):
        return self.T0 + (self.current + 1)*self.dt

    def get_time_step_length(self):
        return self.dt

    def stop(self):
        return self.current >= self.NT - 1

    def advance(self):
        self.current += 1

    def reset(self):
        self.current = 0


class ChebyshevTimeLine():
    def __init__(self, T0, T1, N):
        self.NT = N + 1
        self.T0 = T0
        self.T1 = T1
        self.theta = np.arange(N+1)*np.pi/N
        self.time = 0.5*(T0+T1) - 0.5*(T1 - T0)*np.cos(self.theta)
        self.dt = self.time[1:] - self.time[0:-1]
        self.current = 0

    def diff(self, F):
        d = np.zeros(F.shape, dtype=np.float)
        d[..., 1:] = (F[..., 1:] - F[..., 0:-1])/self.dt
        return d

    def get_next_time(self):
        return self.time[self.current+1]

    def time_integral(self, q):
        N = self.NT - 1
        theta = self.theta
        a = dct(q, type=1)/N
        A = np.zeros(a.shape, dtype=np.float)
        A[:, 0] = (
                a[:, 0] + 0.5*a[:, 1] +
                np.sum(2*a[:, 2:N]/(1 - np.arange(2, N)**2), axis=-1) +
                a[:, -1]/(1 - N**2)
                )
        A[:, 1:N-1] = 0.5*(a[:, 2:N] - a[:, 0:N-2])/np.arange(1, N-1)
        A[:, N-1] = 0.5/(N-1)*(0.5*a[:, N] - a[:, N-2])
        A[:, N] = -0.5/N*a[:, N-1]
        intq = idct(A, type=1)/2 - 0.25*a[:, [-1]]/(N+1)*theta
        intq *= 0.5*(self.time[-1] - self.time[0])
        return intq
    
    def new_time_integral(self, q):
        N = self.NT - 1
        a = dct(q, type=1)/N
        intq = a[:, 0] + np.sum(2*a[:, 2:N:2]/(1 - np.arange(2, N, 2)**2), axis=-1) + a[:, -1]/(1 - N**2)
        intq *= 0.5*(self.time[-1] - self.time[0])
        return intq

    def get_number_of_time_steps(self):
        return self.NT

    def get_current_time_step(self):
        return self.current

    def get_current_time_step_length(self):
        return self.dt[self.current]

    def stop(self):
        return self.current >= self.NT - 1

    def advance(self):
        self.current += 1

    def reset(self):
        self.current = 0
