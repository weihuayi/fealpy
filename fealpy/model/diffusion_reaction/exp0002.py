from typing import Sequence
import sympy as sp

from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike

from ...mesher import BoxMesher2d

class Exp0002(BoxMesher2d):
    def __init__(self):
        """
        PolyData provides data and methods for a 2D elliptic PDE problem with a polynomial exact solution.
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
        self.box = [0, 1, 0, 1]
        super().__init__(box=self.box)
        self.manager, = bm._backends
        x1, x2 = sp.symbols('x1, x2', real=True)
        self.c = 2  # Reaction coefficient
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
        # 计算源项和期望状态
        self.compute_terms()

    def geo_dimension(self) -> int:
        """
        Return the geometric dimension of the problem.

        Returns:
            int: The geometric (spatial) dimension of the domain.
        """
        return 2
    
    def domain(self) -> Sequence[float]:
        """
        Returns the computational domain of the problem as a list of floats.

        The domain is defined by the spatial extent in each coordinate direction.
        For a 2D problem, the returned list typically contains four values: 
        [x_min, x_max, y_min, y_max].

        Returns:
            Sequence[float]: The boundaries of the computational domain.
        """
        return [0, 1, 0, 1]
    
    def compute_terms(self):
        """
        Precomputes and stores key terms required for solving the PDE, including the divergence of the vector field `p` and the source term `f`.
        """
        x1, x2 = self.x1, self.x2
        # 计算 div(p)
        self.div_p = sp.diff(self.p0, x1) + sp.diff(self.p1, x2)
        # 计算源项 f
        self.f = self.div_p + self.c * self.u
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        Compute the exact solution at specified spatial points.

        Parameters:
            p (TensorLike): An array of points in space where the exact solution is to be evaluated.
                The input should be of shape (..., 2), where the last dimension corresponds to the spatial coordinates (x1, x2).

        Returns:
            TensorLike: The computed exact solution values at the given points.
        """
        result = sp.lambdify([self.x1, self.x2], self.u ,self.manager)
        return result(p[...,0], p[...,1])
    
    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """
        Compute the gradient of the solution at specified spatial points.

        Parameters:
            p (TensorLike): An array of spatial points where the gradient is to be evaluated.
                The shape should be (..., 2), where the last dimension corresponds to the spatial coordinates (x1, x2).

        Returns:
            TensorLike: An array of the same shape as `p`, containing the gradient vectors of the solution at each point.
                The last dimension represents the gradient components with respect to x1 and x2.
        """
        result = bm.zeros_like(p)
        du_dx1 = sp.diff(self.u, self.x1)
        du_dx2 = sp.diff(self.u, self.x2)
        result = bm.zeros_like(p)
        du_dx1 = sp.diff(self.u, self.x1)
        du_dx2 = sp.diff(self.u, self.x2)
        grad_u = sp.Matrix([du_dx1, du_dx2])
        grad_u_func = sp.lambdify([self.x1, self.x2], grad_u, self.manager)
        result[..., 0], result[..., 1] = grad_u_func(p[..., 0], p[..., 1])
        return result
    
    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        """
        Compute the flux vector at specified spatial points.
        The flux is defined as the negative product of the diffusion coefficient matrix and the gradient of the solution.

        Parameters:
            p (TensorLike): An array of spatial points where the flux is to be evaluated.
                The shape should be (..., 2), where the last dimension corresponds to the spatial coordinates (x1, x2).
        Returns:
            TensorLike: An array of the same shape as `p`, containing the flux vectors at each point.
                The last dimension represents the flux components corresponding to the diffusion coefficient matrix.
        """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros_like(p)
        p0 = sp.lambdify([self.x1, self.x2], self.p0, self.manager)
        p1 = sp.lambdify([self.x1, self.x2], self.p1, self.manager)
        result[...,0] = p0(x, y) 
        result[...,1] = p1(x, y) 
        return result
    
    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        """
        Compute the diffusion coefficient matrix at specified spatial points.

        This method calculates the diffusion coefficient tensor for each point in the input array `p`.
        The resulting tensor is a 2x2 matrix for each point, where the diagonal entries depend on the
        coordinates: the (0,0) entry is `1 + x**2` and the (1,1) entry is `1 + y**2`. Off-diagonal entries are zero.

        Parameters:
            p (TensorLike): An array of shape (..., 2) representing spatial coordinates, where the last dimension
                corresponds to the x and y coordinates of each point.

        Returns:
            TensorLike: An array of shape (..., 2, 2) containing the diffusion coefficient matrices at each input point.
        """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1+x**2 
        result[..., 1, 1] = 1+y**2
        return result 
    
    def convection_coef(self, p: TensorLike) -> TensorLike:
        """
        Returns the convection coefficient at the given spatial points.

        This method provides the convection coefficient vector field evaluated at the specified points `p`. 
        For this particular problem, the convection coefficient is not defined and thus returns a zero tensor 
        with the same shape as the first component of `p`.

        Parameters:
            p (TensorLike): The spatial points at which to evaluate the convection coefficient. 
                Should be an array-like object where the last dimension corresponds to spatial coordinates.

        Returns:
            TensorLike: A tensor of zeros with the same shape as `p[..., 0]`, representing the convection coefficient.
        """
        return bm.zeros_like(p[..., 0])
    
    def reaction_coef(self, p: TensorLike) -> TensorLike:
        """
        Compute and return the reaction coefficient tensor for the given points.

        Parameters:
            p (TensorLike): The coordinates of the points at which to evaluate the reaction coefficient.
                The input should be a tensor-like object where the last dimension corresponds to spatial coordinates.

        Returns:
            TensorLike: A tensor filled with the reaction coefficient value (2) for each input point,
                matching the shape of the input points except for the coordinate dimension.
        """
        return bm.full(p[..., 0].shape, 2, dtype=self.manager.float_type)
    
    @cartesian
    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        """
        Computes the inverse of the diffusion coefficient matrix at specified spatial points.

        Parameters:
            p (TensorLike): An array of spatial points with shape (..., 2), where the last dimension represents the (x, y) coordinates.

        Returns:
            TensorLike: An array of shape (..., 2, 2) representing the inverse diffusion coefficient matrices evaluated at the input points.
        """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1/(1+x**2)
        result[..., 1, 1] = 1/(1+y**2)
        return result 
    
    @cartesian
    def source(self, p: TensorLike, index=None):
        """
        Compute the source term \( f \) at specified spatial points.

        Parameters:
            p (TensorLike): An array-like object containing the spatial coordinates where the source term is evaluated.
                The last dimension should represent the spatial coordinates (e.g., [x, y]).
            index (optional): An optional index to select specific points or subsets. Defaults to None.

        Returns:
            The computed values of the source term \( f \) at the given spatial points.
        """
        result = sp.lambdify([self.x1, self.x2], self.f, self.manager)
        return result(p[..., 0], p[..., 1])
    
    @cartesian
    def grad_dirichlet(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Compute the gradient of the Dirichlet boundary condition at specified points.

        This method returns the gradient of the Dirichlet condition, which is zero in this case.

        Parameters:
            p (TensorLike): An array of spatial points where the gradient is to be evaluated.
                The shape should be (..., 2), where the last dimension corresponds to the spatial coordinates (x1, x2).
            n (TensorLike): An array representing the normal vectors at the boundary points.
                This parameter is not used in this implementation.
        Returns:
            TensorLike: An array of zeros with the same shape as `p`, representing the gradient of the Dirichlet condition.
        """
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """
        Specifies the Dirichlet boundary condition for the problem, enforcing the solution value to be zero on the boundary.

        Parameters:
            p (TensorLike): The coordinates of the points where the Dirichlet condition is applied.

        Returns:
            TensorLike: An array of zeros with the same shape as the first component of `p`, representing the boundary values.
        """
        return bm.zeros_like(p[..., 0])
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """
        Check if point is on boundary ∂Ω
        This method determines whether the given points `p` lie on the boundary of the domain defined by the box [0, 1] x [0, 1].
        
        Parameters:
            p (TensorLike): An array of shape (..., 2) representing the coordinates of the points to be checked.
        Returns:
            TensorLike: A boolean array of the same shape as `p`, where each element indicates whether the corresponding point is on the boundary.
        """
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(x) < atol) | (bm.abs(x - 1.0) < atol) |
            (bm.abs(y) < atol) | (bm.abs(y - 1.0) < atol)
        )
        return on_boundary