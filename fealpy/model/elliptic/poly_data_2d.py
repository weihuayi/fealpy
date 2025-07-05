from typing import Sequence
from fealpy.decorator import cartesian, variantmethod
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
import sympy as sp
from fealpy.model.box_domain_mesher import BoxDomainMesher2d

class PolyData2D(BoxDomainMesher2d):

    def __init__(self, c=2):
        """PolyData provides data and methods for a 2D elliptic PDE problem with a polynomial exact solution.
        The model problem is:
            -div(A ∇u) + c u = f,   in Ω = [0, 1] x [0, 1]
                ∇u · n = 0,        on ∂Ω (Neumann)
        with the exact solution:
            u(x, y) = x^2 (1-x)^2 y^2 (1-y)^2
        The diffusion coefficient A, reaction coefficient c, and source term f are defined as:
            A = [[1 + x^2, 0], [0, 1 + y^2]]
            c = 2 (default)
            f(x, y) is computed to match the exact solution and coefficients.
        Homogeneous Neumann boundary conditions are imposed on all boundaries.
        This class provides methods for mesh generation, coefficients, exact solution, gradient, flux, and boundary identification for use in finite element simulations.
        """
        self.c = c
        self.manager, = bm._backends

        x1, x2 = sp.symbols('x1, x2', real=True)
        self.x1 = x1
        self.x2 = x2
        self.u = x1**2 * (1-x1)**2 * x2**2 * (1-x2)**2

        # 计算通量变量 p = -A∇u
        dy_dx1 = sp.diff(self.u, x1)
        dy_dx2 = sp.diff(self.u, x2)
        self.p0 = -(1+x1**2) * dy_dx1
        self.p1 = -(1+x2**2) * dy_dx2
        self.p = sp.Matrix([self.p0, self.p1])
        
        
        # 系数矩阵
        self.A00 = 1+x1**2
        self.A11 = 1+x2**2
        self.A = sp.Matrix([[self.A00, 0], [0, self.A11]])
        
        # 预计算源项和期望状态
        self._precompute_terms()
        
    def _precompute_terms(self):
        """Precompute terms for the PDE."""
        x1, x2 = self.x1, self.x2
        
        # 计算 div(p)
        self.div_p = sp.diff(self.p0, x1) + sp.diff(self.p1, x2)
        
        # 计算源项 f
        self.f = self.div_p + self.c * self.u

        
    def geo_dimension(self) -> int:
        """
        Return the geometric dimension of the problem.

        Returns
        int
            The geometric dimension, which is 2 for this problem.
        """
        return 2

    @cartesian
    def domain(self):
        """Return the computational domain of the problem."""
        return [0, 1, 0, 1]
    
    @variantmethod('tri')
    def init_mesh(self, nx=10, ny=10):
        from fealpy.mesh import TriangleMesh
        d = self.domain()
        mesh = TriangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh
    
    @init_mesh.register('quad')
    def init_mesh(self, nx=10, ny=10):
        from fealpy.mesh import QuadrangleMesh
        d = self.domain()
        mesh = QuadrangleMesh.from_box(d, nx=nx, ny=ny)
        return mesh
    
    @cartesian
    def solution(self, space):
        """ Compute the exact solution y at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.u ,self.manager)
        return result(space[...,0], space[...,1])
    
    @cartesian
    def flux(self, space):
        """ Compute the flux p at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros_like(space)
        p0 = sp.lambdify([self.x1, self.x2], self.p0, self.manager)
        p1 = sp.lambdify([self.x1, self.x2], self.p1, self.manager)
        result[...,0] = p0(x, y) 
        result[...,1] = p1(x, y) 
        return result
    
    
    @cartesian
    def diffusion_coef(self, space): 
        """ Compute the diffusion coefficient matrix at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1+x**2 
        result[..., 1, 1] = 1+y**2
        return result 
    
    @cartesian
    def diffusion_coef_inv(self, space):
        """ Compute the inverse of the diffusion coefficient matrix at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1/(1+x**2)
        result[..., 1, 1] = 1/(1+y**2)
        return result 
    
    @cartesian
    def source(self, space, index=None):
        """ Compute the source term f at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.f, self.manager) 
        return result(space[...,0], space[...,1])
    
    @cartesian
    def grad_dirichlet(self, p, space):
        return bm.zeros_like(p[..., 0])
    