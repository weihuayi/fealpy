
from ..backend import backend_manager as bm
bm.set_backend('pytorch')

from ..typing import TensorLike
from ..decorator import cartesian

def bessel_function(v: int, x: TensorLike) -> TensorLike:
    """
    Compute the Bessel function of the first kind of order v at points x.

    Parameters:
        v: Order of the Bessel function (only 0 and 1 are currently supported).
        x: Input values where the Bessel function is evaluated. Can be a tensor,
           array, or scalar depending on the backend.

    Returns:
        TensorLike: Values of the Bessel function at points x.

    Note:
        - For PyTorch backend: Uses native torch.special.bessel_j0 and bessel_j1 functions.
        - For Numpy backends: Uses scipy.special.jv function.
        - Currently only supports orders 0 and 1 for both backends.
    """
    if bm.backend_name == 'pytorch':
        import torch
        if v == 0:
            return torch.special.bessel_j0(x)
        elif v == 1:
            return torch.special.bessel_j1(x)
        else:
            raise NotImplementedError("PyTorch only supports Bessel functions of order 0 and 1.")
    else:
        from scipy.special import jv
        if v == 0:
            return jv(0, x)
        elif v == 1:
            return jv(1, x)
        else:
            raise NotImplementedError("Just supports Bessel functions of order 0 and 1.")

class HelmholtzData2d():
    """
    A class representing the 2D Helmholtz equation with methods for solution, gradient,
    source term, PDE formulation, and Robin boundary conditions.

    Parameters:
        k: Wave number for the Helmholtz equation.
        backend (str): Computational backend ('pytorch', 'numpy'). 
                                Defaults to 'pytorch'.
    """
    def __init__(self, k=1.0):
        self.k = bm.tensor(k, dtype=bm.float64)  # 0 维张量
        c1 = bm.cos(self.k) + bm.sin(self.k)*1j
        c2 = bessel_function(v=0, x=self.k) + 1j * bessel_function(v=1, x=self.k)
        self.c = c1 / c2    # 方程中的常数

    def domain(self):
        return (-0.5, 0.5, -0.5, 0.5)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """
        The exact solution of the 2D Helmholtz equation:
        u(x,y) = (cos(k*r) - c * J0(k*r)) / k,
        where r = sqrt(x^2 + y^2),c = (cos(k) + i*sin(k)) / (J0(k) + i*J1(k)),
        J0 and J1 are Bessel functions of the first kind of order 0 and 1 respectively.

        Parameters:
            p: Input tensor representing spatial coordinates (2D).

        Returns:
            Tensor: The exact solution at given points.
        """
        if bm.backend_name == 'pytorch':
            x = p[..., 0:1]
            y = p[..., 1:2]
        elif bm.backend_name == 'numpy':
            x = p[..., 0]
            y = p[..., 1]
        r = bm.sqrt(x ** 2 + y ** 2)
        val = bm.zeros(x.shape, dtype=bm.complex128)
        val[:] = (bm.cos(self.k * r) - self.c * bessel_function(v=0, x=self.k * r)) / self.k
        return val
    
    def solution_numpy_real(self, p: TensorLike) -> TensorLike:
        """
        The real part of the exact solution, with numpy array input/output.

        Parameters:
            p: Input numpy array representing spatial coordinates.

        Returns:
            NDArray: The real part of the solution at given points.
        """
        sol = self.solution(bm.tensor(p))
        real_ = bm.real(sol)
        return real_.detach().numpy()

    @cartesian
    def solution_numpy_imag(self, p: TensorLike) -> TensorLike:
        """
           The imaginary part of the exact solution, with numpy array input/output.

           Parameters:
               p: Input numpy array representing spatial coordinates.

           Returns:
               NDArray: The imaginary part of the solution at given points.
           """
        sol = self.solution(bm.tensor(p))
        imag_ = bm.imag(sol)
        return imag_.detach().numpy()

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """
        The source term of the 2D Helmholtz equation:
        f(x,y) = sin(k*r) / r, where r = sqrt(x^2 + y^2).

        Parameters:
            p: Input tensor representing spatial coordinates.

        Returns:
            Tensor: The source term at given points.
        """
        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)
        f = bm.zeros(x.shape, dtype=bm.complex128)
        f[:] = bm.sin(self.k * r) / r  # 源项
        return f

    @cartesian
    def gradient(self, p:  TensorLike) -> TensorLike:
        """
        The gradient of the exact solution of the 2D Helmholtz equation:
        ∇u = (-cos(k*r) + c * J1(k*r)) / k * (x/r, y/r),
        where r = sqrt(x^2 + y^2),c = (cos(k) + i*sin(k)) / (J0(k) + i*J1(k)),
        J0 and J1 are Bessel functions of the first kind of order 0 and 1 respectively.

        Parameters:
            p: Input tensor representing spatial coordinates.

        Returns:
            Tensor: The gradient of the solution at given points.
        """
        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)

        val = bm.zeros(p.shape, dtype=bm.complex128)
        u_r = self.c * bessel_function(v=1, x=self.k * r) - bm.sin(self.k * r)
        val[..., 0:1] = u_r * x / r
        val[..., 1:2] = u_r * y / r
        return val

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        """
        Compute the Robin boundary condition for the Helmholtz equation.

        Parameters:
            p: Input tensor representing spatial coordinates with shape (NC, NQ, dof_numel), 
            n: Normal vectors at each point with shape (N, 2), corresponding to p.

        Returns:
            TensorLike: The Robin boundary condition values at given points with shape (N, 1).
                    Computed as ∂u/∂n + iκu, where κ is the wavenumber.
        """
        kappa = 0.0 + self.k * 1j
        if bm.backend_name == 'pytorch':
            val = (self.gradient(p) * n).sum(dim=-1, keepdim=True) + kappa * self.solution(p)
        elif bm.backend_name == 'numpy':
            x = p[..., 0]
            y = p[..., 1]
            grad = self.gradient(p)     # (NC, NQ, GD)
            val = bm.sum(grad*n[:, None, :], axis=-1) # (N,1)
            val += kappa*self.solution(p) 
        return val
    
    def symbolic_com(self):
        """
        Symbolic computation of the solution and its derivatives using SymPy.
        Used for verification and debugging purposes.

        Computes:
        1. The exact solution u in symbolic form
        2. The source term f = -Δu - k²u
        3. Partial derivatives ∂u/∂x and ∂u/∂y

        Prints:
        - The symbolic expression for the source term f
        - The symbolic expressions for ∂u/∂x and ∂u/∂y
        """
        import sympy as sp
        x, y, k , R= sp.symbols('x, y, k, R', real=True)
        r = sp.sqrt(x**2 + y**2)
        J0k = sp.besselj(0, k)
        J1k = sp.besselj(1, k)
        J0kr = sp.besselj(0, k*r)
        u = sp.cos(k*r)/k - J0kr*(sp.cos(k) + sp.I*sp.sin(k))/(J0k + sp.I*J1k)/k
        f = (-u.diff(x, 2) - u.diff(y, 2) - k**2*u).simplify().subs({r:R})
    
        print("f:", f)
        print(u.diff(x).subs({r:R}))
        print(u.diff(y).subs({r:R}))

