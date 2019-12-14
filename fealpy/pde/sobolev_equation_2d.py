import numpy as np

from fealpy.timeintegratoralg.timeline_new import UniformTimeLine

class PolyExpData:
    def __init__(self, nu=1, epsilon=1):
        self.nu = nu
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

    def flux(self, p, t):
        nu = self.nu
        epsilon = self.epsilon
        val = (epsilon - nu)*self.gradient(p, t)
        return val

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

class SinSinExpData:
    def __init__(self, nu=1, epsilon=1):
        self.nu = nu
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
        nu = self.nu
        epsilon = self.epsilon
        val = (epsilon - nu)*self.gradient(p, t)
        return val

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi

        nu = self.nu
        epsilon = self.epsilon
        f = (-2*pi**2*nu + 2*pi**2*epsilon - 1)*exp(-t)*sin(pi*x)*sin(pi*y)
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

        cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n)
        return mesh
