import numpy as np

from ..timeintegratoralg.timeline import UniformTimeLine
from ..mesh import TriangleMesh

class SinSinExpData:
    def __init__(self, mu=1, epsilon=1):
        self.mu = mu
        self.epsilon = epsilon

    def domain(self):
        return [0, 1, 0, 1]

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.exp(-t)
        return u

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(-t)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(-t)
        return val

    def laplace(self, p, t):
        val = -2*np.pi**2*self.solution(p, t)
        return val

    def flux(self, p, t):
        mu = self.mu
        epsilon = self.epsilon
        val = (epsilon - mu)*self.gradient(p, t)
        return val

    def div_flux(self, p, t):
        mu = self.mu
        epsilon = self.epsilon
        return (epsilon - mu)*self.laplace(p, t)

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi

        mu = self.mu
        epsilon = self.epsilon
        f = (-2*pi**2*mu + 2*pi**2*epsilon - 1)*np.exp(-t)*np.sin(pi*x)*np.sin(pi*y)
        return f

    def dirichlet(self, p, t):
        return self.solution(p, t)

    def init_value(self, p):
        return self.solution(p, 0.0)

    def time_mesh(self, t0, t1, NT):
        return UniformTimeLine(t0, t1, NT)

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        if meshtype is 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
        return mesh

class PolyExpData:
    def __init__(self, mu=1, epsilon=1):
        self.mu = mu
        self.epsilon = epsilon

    def domain(self):
        return [0, 1, 0, 1]

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = x*(1-x)*y*(1-y)*np.exp(-t)
        return u

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = (1-2*x)*y*(1-y)*np.exp(-t)
        val[..., 1] = (1-2*y)*x*(1-x)*np.exp(-t)
        return val

    def flux(self, p, t):
        mu = self.mu
        epsilon = self.epsilon
        val = (epsilon - mu)*self.gradient(p, t)
        return val

    def div_flux(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        mu = self.mu
        epsilon = self.epsilon
        val = 2*(epsilon - mu)*(x*(x - 1) + y*(y - 1))*np.exp(-t)
        return val


    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]

        mu = self.mu
        epsilon = self.epsilon
        t0 = np.exp(-t)
        t1 = -x*y*(-x + 1)*(-y + 1) - (epsilon - mu)*(2*x*(x - 1) + 2*y*(y - 1))
        return t0*t1

    def dirichlet(self, p, t):
        return self.solution(p, t)

    def init_value(self, p):
        return self.solution(p, 0.0)

    def time_mesh(self, t0, t1, NT):
        return UniformTimeLine(t0, t1, NT)

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh

