import numpy as np

from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.mesh.Tritree import Tritree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh

from fealpy.timeintegratoralg.timeline import UniformTimeLine
from fealpy.timeintegratoralg.timeline import ChebyshevTimeLine

class SinSinExpData:
    def __init__(self):
        self.diffusion_coefficient = 1

    def domain(self):
        return [0, 1, 0, 1]
    def time_mesh(self, t0, t1, NT, timeline='uniform'):
        if timeline is 'uniform':
            return UniformTimeLine(t0, t1, NT)
        elif timeline is 'chebyshev':
            return ChebyshevTimeLine(t0, t1, NT)

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

    def init_value(self, p):
        return self.solution(p, 0.0)

    def diffusion_coefficient(self):
        return self.diffusionCoefficient

    def solution(self, p, t):
        """ The exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)*np.exp(-t)
        return u

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        return (2*np.pi**2-1)*self.solution(p, t)

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return self.solution(p, t)

class SinCosExpData:
    def __init__(self):
        self.diffusion_coefficient = 1.0/3.0

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

    def init_value(self, p):
        return self.solution(p, 0.0)

    def diffusion_coefficient(self):
        return self.diffusionCoefficient

    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[:, 0]
        y = p[:, 1]
        pi = np.pi
        u = np.sin(pi*x)*np.cos(pi*y)*np.exp(-pi**2/8*t)
        return u

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        return 0.0

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return 0.0 

    def neuman(self, p, t):
        """ Neuman boundary condition
        """
        return 0.0

    def is_dirichlet_boundary(self, p):
        eps = 1e-14 
        return (p[:, 0] < eps) | (p[:, 0] > 1.0 - eps) 

    def is_neuman_boundary(self, p):
        eps = 1e-14
        return (p[:, 1] < eps) | (p[:, 1] > 1.0 - eps)


class SpaceMeasureDiracSourceData:

    def __init__(self):
        pass

    def domain(self):
        points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return points, facets

    def time_mesh(self, t0, t1, N):
        return UniformTimeLine(t0, t1, N)

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)

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

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        """ The exact solution
        """
        pi = np.pi
        cos = np.cos
        sin = np.sin

        gt = 1 - np.exp(-500*(t - 0.5)**2)
        p0 = 0.4*np.array([cos(2*pi*t), sin(2*pi*t)], dtype=p.dtype)
        r = np.sqrt(np.sum((p - p0)**2, axis=-1))
        val = -0.5*np.log(r)*gt/pi
        return val

    def gradient(self, p, t):

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        cos = np.cos
        sin = np.sin

        gt = 1 - np.exp(-500*(t - 0.5)**2)
        p0 = 0.4*np.array([cos(2*pi*t), sin(2*pi*t)], dtype=p.dtype)
        x = p - p0
        r = -1/(2*pi*np.sum(x**2, axis=-1))*gt

        val[..., 0] = r*gt*x[..., 0]
        val[..., 1] = r*gt*x[..., 1]

#        r = -1/(2*pi*np.sum(g**2, axis=-1))*gt
#        val *= r[..., np.newaxis]
        return val

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        pi = np.pi
        cos = np.cos
        sin = np.sin

        gt = -250*(2*t - 1.0)*np.exp(-500*(t - 0.5)^2)/pi
        p0 = 0.4*np.array([cos(2*pi*t), sin(2*pi*t)], dtype=p.dtype)
        r = np.sqrt(np.sum((p - p0)**2, axis=-1))
        val = gt*np.log(r)
        return val

    def dirac_source(self, p, t):
        pi = np.pi
        cos = np.cos
        sin = np.sin
        gt = 1 - np.exp(-500*(t - 0.5)^2)
        p0 = 0.4*np.array([cos(2*pi*t), sin(2*pi*t)], dtype=p.dtype)
        return gt, p0

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return 0.0

    def neuman(self, p, t):
        """ Neuman boundary condition
        """
        return 0.0

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        x = p[..., 0]
        y = p[..., 1]
        return (x < eps) | (x > 1.0 - eps) | (y < eps) | (z > 1.0 - eps)


class TimeMeasureDiracSourceData:
    def __init__(self):
        pass

    def domain(self):
        points = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        facets = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return points, facets

    def time_mesh(self, t0, t1, N): ##???
        return UniformTimeLine(t0, t1, N)

    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float)

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

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        """ The exact solution
        """
        if t < 0.5:
            t0 = t**2
        else:
            t0 = t**2 + 2*t
        
        r = np.sum((p - (t - 0.5))**2, axis=-1)
        val = 0.1*np.exp(-25*r)*t0
        return val


    def gradient(self, p, t):

        val = np.zeros(p.shape, dtype=p.dtype)
        pi = np.pi
        x = p - (t - 0.5)
        gt = 0.1*np.exp(-25*np.sum(x**2, axis=-1))
        g0 = -25/(np.sum(x**2, axis=-1))

        if t < 0.5:
            val[..., 0] = g0*gt*x[..., 0]*t**2
            val[..., 1] = g0*gt*x[..., 1]*t**2
        else:
            val[..., 0] = g0*gt*x[..., 0]*(t**2 + 2*t)
            val[..., 1] = g0*gt*x[..., 1]*(t**2 + 2*t)

        return val

    def source(self, p, t):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        val = 0.0
        return val

    def dirac_source(self, p, t):

        gt = 0.1*np.exp(-25*np.sum((p - (t - 0.5))**2, axis=-1))
        p0 = 1/2 
        return gt, p0

    def dirichlet(self, p, t):
        """ Dilichlet boundary condition
        """
        return 0.0

    def neuman(self, p, t):
        """ Neuman boundary condition
        """
        return 0.0

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        x = p[..., 0]
        y = p[..., 1]
        return (x < eps) | (x > 1.0 - eps) | (y < eps) | (z > 1.0 - eps)



