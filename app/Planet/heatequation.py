import numpy as np
from fealpy.geometry.implicit_surface import SphereSurface
from fealpy.mesh.simple_mesh_generator import unitcircledomainmesh
from fealpy.mesh.SurfaceTriangleMeshOptAlg import SurfaceTriangleMeshOptAlg 
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh, LagrangeWedgeMesh
#from LagrangeWedgeMesh import LagrangeWedgeMesh
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian, barycentric

class CosCosCosExpData:
    def __init__(self):
        pass

    def init_mesh(self, n=0, h=0.1, nh=5, p=1):
        print('p:', p)
        surface = SphereSurface()
        mesh = surface.init_mesh(meshtype='tri', p=p)

#        node = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
#                dtype=np.float)
#        cell = np.array([[0, 1, 2], [3, 0, 2]], dtype=np.int)
#        mesh = LagrangeTriangleMesh(node, cell, p=p)

        mesh.uniform_refine(n)
        mesh = LagrangeWedgeMesh(mesh, h, nh, p=p)
        self.mesh = mesh
        return mesh

    @cartesian    
    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        u = np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z)*np.exp(-t)
        return u
    
    @cartesian    
    def nonlinear(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        u = np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z)*np.exp(-t)
        return u**4
    
    @cartesian    
    def gradient(self, p, t):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)*np.cos(pi*z)*np.exp(-t)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)*np.cos(pi*z)*np.exp(-t)
        val[..., 2] = -pi*np.cos(pi*x)*np.cos(pi*y)*np.sin(pi*z)*np.exp(-t)
        return val

    @cartesian    
    def source(self, p, t):
        """ The right hand side of Heat equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = (3*pi**2-1)*np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z)*np.exp(-t)
        return val

    def neumann(self, p, n, t0):
        grad = self.gradient(p, t0)
        val = np.sum(grad*n, axis=-1)
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        tface, qface = self.mesh.entity('face')
        nf = len(tface)
        boundary_dir_tface_index = np.zeros(nf, dtype=np.bool_)
        boundary_dir_tface_index[-nf//2:] = True
        return boundary_dir_tface_index 

    def dirichlet(self, p, t):
        return self.solution(p, t)

    @cartesian
    def is_dirichlet_boundary(self, p):
        tface, qface = self.mesh.entity('face')
        nf = len(tface)
        boundary_dir_tface_index = np.zeros(nf, dtype=np.bool_)
        boundary_dir_tface_index[-nf//2:] = True
        return boundary_dir_tface_index 

    @cartesian    
    def robin(self, p, n, t1, t0):
        """ Robin boundary condition
        """
        grad = self.gradient(p, t1) + self.gradient(p, t0)# (NQ, NF, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p, t1) + self.solution(p, t0)
        return val, kappa
    
    @cartesian
    def is_robin_boundary(self):
        tface, qface = self.mesh.entity('face')
        nf = len(tface)
        boundary_robin_tface_index = np.zeros(nf, dtype=np.bool_)
        boundary_robin_tface_index[:nf//2] = True
        return boundary_robin_tface_index 

    @cartesian    
    def boundary_robin(self, p, n, t0, t):
        grad = self.gradient(p, t) + self.gradient(p, t0)# (NQ, NF, 3)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.nonlinear(p, t) + self.nonlinear(p, t0)
        return val, kappa
    
    def boundary_nonlinear_robin(self, p, n, t):
        grad = self.gradient(p, t) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.nonlinear(p, t)
        return val, kappa

class CosCosExpData:
    def __init__(self, h):
        self.h = h

    def init_mesh(self, n=0):
        mesh = unitcircledomainmesh(self.h) 
        mesh.uniform_refine(n)
#        node = np.array([[0,0], [1,0], [0,1]], dtype=np.float)
#        cell = np.array([[0, 1, 2]])
#        mesh = TriangleMesh(node, cell)
        return mesh

    @cartesian    
    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.cos(pi*x)*np.cos(pi*y)*np.exp(-t)
        return u
    
    @cartesian    
    def nonlinear(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.cos(pi*x)*np.cos(pi*y)*np.exp(-t)
        return u**4
    
    @cartesian    
    def gradient(self, p, t):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(-t)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(-t)
        return val

    @cartesian    
    def source(self, p, t):
        """ The right hand side of Heat equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (2*pi**2-1)*np.cos(pi*x)*np.cos(pi*y)*np.exp(-t)
        return val
    
    def dirichlet(self, p, t):
        return self.solution(p, t)

    @cartesian    
    def robin(self, p, n, t1, t0):
        """ Robin boundary condition
        """
        grad = self.gradient(p, t1) + self.gradient(p, t0)# (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p, t1) + self.solution(p, t0)
        return 0.5*val, kappa
    
    @cartesian    
    def boundary_robin(self, p, n, t0, t):
        grad = self.gradient(p, t) + self.gradient(p, t0)# (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.nonlinear(p, t) + self.nonlinear(p, t0)
        return val, kappa
    
    def boundary_nonlinear_robin(self, p, n, t):
        grad = self.gradient(p, t) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.nonlinear(p, t)
        return val, kappa

class XYTData:
    def __init__(self, h):
        self.h = h

    def init_mesh(self, n=0):
        mesh = unitcircledomainmesh(self.h) 
        mesh.uniform_refine(n)
#        node = np.array([[0,0], [1,0], [0,1]], dtype=np.float)
#        cell = np.array([[0, 1, 2]])
#        mesh = TriangleMesh(node, cell)
        return mesh

    def solution(self, p, t):
        """ The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = x*y*t
        return u
    
    def gradient(self, p, t):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = y*t
        val[..., 1] = x*t
        return val
    
    def source(self, p, t):
        """ The right hand side of Heat equation
        INPUT:
            p: array object, N*2 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = x*y
        return val

    def robin(self, p, n, t0, t1):
        """ Robin boundary condition
        """
        grad = self.gradient(p, t0)+ self.gradient(p, t1) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p, t0) +self.solution(p, t1)
        return val, kappa


