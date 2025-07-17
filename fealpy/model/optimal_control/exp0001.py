from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike
import sympy as sp
from ..mesher.box_mesher import BoxMesher2d

class Exp0001(BoxMesher2d):
    """
    A PDE example with homogeneous Neumann boundary conditions on the unit square.

    This class defines the exact solution, flux variables, and coefficients for a specific PDE problem.
    It provides methods to compute the source term, flux, expected state, and other related quantities.
    The exact solution is constructed to satisfy homogeneous Neumann boundary conditions.

    Parameters
    c : float, optional, default=1
        Problem parameter used in the source term and expected state.

    Attributes
        c : float
            Problem parameter.
        manager : object
            Backend manager for numerical computation.
        x1, x2 : sympy.Symbol
            Symbolic variables for spatial coordinates.
        y, z, u : sympy.Expr
            Symbolic expressions for the exact state, adjoint state, and control.
        p, q : sympy.Matrix
            Symbolic flux and adjoint flux.
        A : sympy.Matrix
            Coefficient matrix.
        f : sympy.Expr
            Source term.
        y_d : sympy.Expr
            Desired state.
        div_p, div_q : sympy.Expr
            Divergence of flux and adjoint flux.

    Methods
        domain()
            Return the computational domain.
        y_solution(space)
            Evaluate the exact state at given points.
        z_solution(space)
            Evaluate the adjoint state at given points.
        u_solution(space)
            Evaluate the control at given points.
        p_solution(space)
            Evaluate the flux at given points.
        q_solution(space)
            Evaluate the adjoint flux at given points.
        f_fun(space)
            Evaluate the source term at given points.
        y_d_fun(space)
            Evaluate the desired state at given points.

    Notes
        The problem is defined on [0, 1] x [0, 1] with variable coefficients and homogeneous Neumann boundary conditions.
        The class is designed for use with the FEALPy finite element library.

    Examples
        >>> example = Example1(c=1)
        >>> import numpy as np
        >>> pts = np.array([[0.5, 0.5]])
        >>> y_val = example.y_solution(pts)
        >>> print(y_val)
    """
    
    def __init__(self, options: dict={}): 
        '''
        """
        Initialize the OptimalControlData class.

        This constructor sets up the symbolic variables, exact solutions, fluxes, coefficient matrices,
        and precomputes the source term and desired state for the PDE optimal control problem.

        Parameters
        c : float, optional, default=1
            Problem parameter used in the source term and expected state.

        Returns
        None

        Notes
        The symbolic expressions are constructed using sympy for later numerical evaluation.
        """
        '''
        self.box = [0.0, 1.0, 0.0, 1.0]
        super().__init__(box=self.box)
        self.c = options.get('c', 1)
        self.manager, = bm._backends

        x1, x2 = sp.symbols('x1, x2', real=True)
        self.x1 = x1
        self.x2 = x2

        # 修改为满足齐次Neumann边界条件的精确解
        self.y = x1**2 * (1-x1)**2 * x2**2 * (1-x2)**2
        self.z = -self.y
        self.u = self.y

        # 计算通量变量 p = -A∇y
        dy_dx1 = sp.diff(self.y, x1)
        dy_dx2 = sp.diff(self.y, x2)
        self.p0 = -(1+x1**2) * dy_dx1
        self.p1 = -(1+x2**2) * dy_dx2
        self.p = sp.Matrix([self.p0, self.p1])
        
        # 伴随通量 q = A∇z = -p
        self.q0 = -self.p0
        self.q1 = -self.p1
        self.q = sp.Matrix([self.q0, self.q1])
        
        # 系数矩阵
        self.A00 = 1+x1**2
        self.A11 = 1+x2**2
        self.A = sp.Matrix([[self.A00, 0], [0, self.A11]])
        
        # 预计算源项和期望状态
        self._precompute_terms()
        
    def _precompute_terms(self):
        '''
        Precomputes and assigns intermediate symbolic expressions for the optimal control problem.
        This method calculates and stores several symbolic terms required for the optimal control model, 
        including the divergence of vector fields `p` and `q`, the source term `f`, and the desired state `y_d`. 
        The computations depend on the class attributes and symbolic variables.
        Parameters
            None
        Returns
            None
        Notes
            The computed attributes are:
                - self.div_p: Divergence of vector field p = (p0, p1)
                - self.f: Source term, determined by the value of self.c
                - self.div_q: Divergence of vector field q = (q0, q1)
                - self.y_d: Desired state, computed from y, z, c, and div_q
            Symbolic differentiation is performed using SymPy (sp).
        '''
        x1, x2 = self.x1, self.x2
        
        # 计算 div(p)
        self.div_p = sp.diff(self.p0, x1) + sp.diff(self.p1, x2)
        
        # 计算源项 f
        if self.c == 1:
            self.f = self.div_p
        else:
            self.f = self.div_p - self.u
            
        # 计算 div(q)
        self.div_q = sp.diff(self.q0, x1) + sp.diff(self.q1, x2)
        
        # 计算期望状态 y_d
        self.y_d = self.y-self.c*self.z - self.div_q
        
    def geo_dimension(self) -> int:
        """
        Return the geometric dimension of the problem.

        Returns
        int
            The geometric dimension, which is 2 for this problem.
        """
        return 2

    def domain(self):
        return [0, 1, 0, 1]
    
    @cartesian
    def y_solution(self, space):
        """ Compute the exact solution y at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.y, self.manager)
        return result(space[...,0], space[...,1])
    
    @cartesian
    def z_solution(self, space):
        """ Compute the exact solution z at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.z, self.manager)
        return result(space[...,0], space[...,1])
    
    @cartesian
    def u_solution(self, space):
        """ Compute the control u at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.u, self.manager)
        return result(space[...,0], space[...,1])
    
    @cartesian
    def p_solution(self, space):
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
    def q_solution(self, space):
        """ Compute the adjoint flux q at given points in space."""
        # q = -p
        return -self.p_solution(space)
    
    @cartesian
    def source_average(self, space):
        result = sp.lambdify([self.x1, self.x2], self.c, self.manager) 
        return result(space[...,0], space[...,1])
    
    @cartesian
    def A_matirx(self, space): 
        """ Compute the coefficient matrix A at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1+x**2 
        result[..., 1, 1] = 1+y**2
        return result 
    
    @cartesian
    def A_inverse(self, space):
        """ Compute the inverse of the coefficient matrix A at given points in space."""
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1/(1+x**2)
        result[..., 1, 1] = 1/(1+y**2)
        return result 
    
    @cartesian
    def f_fun(self, space, index=None):
        """ Compute the source term f at given points in space."""
        result = sp.lambdify([self.x1, self.x2], self.f, self.manager) 
        return result(space[...,0], space[...,1])
    
    @cartesian
    def pd_fun(self, space):
        return self.p_solution(space)
    
    @cartesian
    def div_pd_fun(self, space):
        result = sp.lambdify([self.x1, self.x2], self.div_p, self.manager) 
        return result(space[...,0], space[...,1])
    
    @cartesian
    def yd_fun(self, space):
        result = sp.lambdify([self.x1, self.x2], self.y_d, self.manager) 
        return result(space[...,0], space[...,1])
    
    @cartesian
    def grad_dirichlet(self, p, space):
        return bm.zeros_like(p[..., 0])
    
