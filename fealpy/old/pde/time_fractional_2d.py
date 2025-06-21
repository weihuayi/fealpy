import numpy as np
from scipy.special import gamma, beta

from ..mesh.TriangleMesh import TriangleMesh
from ..mesh.Quadtree import Quadtree
from ..mesh.QuadrangleMesh import QuadrangleMesh
from ..mesh.Tritree import Tritree
from ..mesh.StructureQuadMesh import StructureQuadMesh

from ..timeintegratoralg.timeline import UniformTimeLine


class FisherData2d:
    def __init__(self):
        self.alpha = 1.0/3.0
        self.nu = 2.0

    def time_mesh(self, t0, t1, N):
        return UniformTimeLine(t0, t1, N)

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
        elif meshtype is 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype is 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)
            mesh = Tritree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def diffusion_coefficient(self):
        return 1.0

    def reaction_coefficient(self):
        return 1.0

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        """ The exact solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos

        nu = self.nu
        alpha = self.alpha

        x = p[:, 0]
        y = p[:, 1]

        u = t**nu*sin(2*pi*x)*sin(2*pi*y)
        return u

    def gradient(self, p):
        """ The gradient of the solution
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos

        nu = self.nu
        alpha = self.alpha

        x = p[..., 0]
        y = p[..., 1]
        theta = np.arctan2(y, x)
        theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
        r = x**2 + y**2
        shape = p.shape[:-1] + (2, 2)
        m = np.zeros(shape, dtype=p.dtype)
        m[..., 0, 0] = cos(theta)
        m[..., 0, 1] = r*sin(theta)
        m[..., 1, 0] = sin(theta)
        m[..., 1, 1] = r*cos(theta)
        m = np.linalg.inv(m)
        val = np.zeros(p.shape, dtype=p.dtype)
        z_r = (2/3*r**(-1/3) - 5/2*r**(3/2))*sin(2/3*theta)
        z_t = 2/3*(r**(2/3) - r**2.5)*cos(2/3*theta)
        val[..., 0] = m[..., 0, 0]*z_r + m[..., 0, 1]*z_t
        val[..., 1] = m[..., 1, 0]*z_r + m[..., 1, 1]*z_t
        return val

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        pi = np.pi
        sin = np.sin
        cos = np.cos
        c = (8*pi**2 + 1)*t**nu + nu*t**(nu-alpha)/gamma(1-alpha)*beta(nu, 1 - alpha)
        val = sin(2*pi*x)*sin(2*pi*y)
        val = c*val + (t**nu*val)**2
        return val

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return 0.0

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[:, 0] < eps) | (p[:, 0] > 1.0 - eps) | (p[:, 1] < eps) | (p[:, 1] > 1.0 - eps)
