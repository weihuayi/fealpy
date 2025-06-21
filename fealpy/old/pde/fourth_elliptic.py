
import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.mesh.Tritree import Tritree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d


class CosData:
    """
    -\Delta^2 u - u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)


    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val


class PolynomialData:
    def __init__(self):
        pass

    def solution(self, p):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        u = (x-x**2)*(y-y**2)
        return u

    def init_mesh(self, n=4, meshtype='tri', h=0.1):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        if meshtype == 'quad':
            node = np.array([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (0.5, 0),
                (1, 0.4),
                (0.3, 1),
                (0, 0.6),
                (0.5, 0.45)], dtype=np.float64)
            cell = np.array([
                (0, 4, 8, 7), (4, 1, 5, 8),
                (7, 8, 6, 3), (8, 5, 2, 6)], dtype=np.int_)
            mesh = QuadrangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'halfedge':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'squad':
            mesh = StructureQuadMesh([0, 1, 0, 1], h)
            return mesh
        else:
            raise ValueError("".format)

    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        rhs = 2*(y-y**2)+2*(x-x**2)+8
        return rhs


    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)

    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        uprime = np.zeros(p.shape, dtype=np.float64)
        uprime[..., 0] = (1-2*x)*(y-y**2)
        uprime[..., 1] = (1-2*y)*(x-x**2)
        return uprime

class LShapeRSinData:
    def __init__(self, eps):
        self.eps = eps

    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (0, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)], dtype=np.float64)
        if meshtype == 'tri':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([
                (0, 1, 3, 2),
                (2, 3, 6, 5),
                (3, 4, 7, 6)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([
                (1, 3, 0),
                (2, 0, 3),
                (3, 6, 2),
                (5, 2, 6),
                (4, 7, 3),
                (6, 3, 7)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))

    def domain(self):
        points = [[0, 0], [1, 0], [1, 1], [-1, 1], [-1, -1], [0, -1]]
        facets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        return points, facets

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.sin(pi*y)
        return u

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = (4*self.eps*pi**4)*np.sin(pi*x)*np.sin(pi*y)
        return u

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        sin = np.sin
        cos = np.cos
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = pi*cos(pi*x)*sin(pi*y)
        val[..., 1] = pi*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

class SqrtData():
    def __init__(self, k = 3):
        self.k = k

    def init_mesh(self, n=4, meshtype='tri'):
        node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
        if meshtype=='tri':
            cell = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.ds.NV = 3
            mesh.init_level_info()
        elif meshtype == 'quad':
            cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            mesh = HalfEdgeMesh2d.from_mesh(mesh)
            mesh.init_level_info()
        elif meshtype == 'tritree':
            cell = np.array([
                (0, 1, 2),
                (2, 3, 0)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
        return mesh

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        k = self.k
        z = x**2+y**2
        return z**k

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        x = p[..., 0]
        y = p[..., 1]
        z = x**2+y**2
        k = self.k
        F = 16*(k**2)*((k-1)**2)*(z**(k-2))-4*(k**2)*(z**(k-1))
        return F

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = x**2+y**2
        k = self.k
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 2*k*x*z**(k-1)
        val[..., 1] = 2*k*y*z**(k-1)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

