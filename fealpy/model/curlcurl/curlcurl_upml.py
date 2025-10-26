from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm

class CurlCurlUPML:
    """
    Unified treatment of 2D and 3D UPML layers in the Maxwell equations:

        curl (mu^{-1} · A · curl E) -ω^2 · ε · A^{-1} · E = f

    where:
        A = |F|^{-1} · Fᵗ · F, with F being the Jacobian matrix of the complex coordinate stretching.

    In the UPML, F is a diagonal matrix:
        - 2D: F = diag(s1(x), s2(y))
        - 3D: F = diag(s1(x), s2(y), s3(z))

    The stretching function:
        s_j(x_j) = 1 + i · σ_j(x_j)

    The absorption profile σ_j(t) is defined as:
        σ_j(t) = σ_max · ((t - a)/δ)^p     (only within the PML region)
    """

    def __init__(self, omega, mu, epsilon,
                 limits, delta, 
                 sigma_max=8.0, pl=5, dim=3, **kwargs):
        """
        Initialize UPML parameters.

        Parameters
        ----------
        omega : float
            Angular frequency.
        mu : float or array_like
            Relative magnetic permeability.
        epsilon : float or array_like
            Relative permittivity.
        k : float
            Wave number.
        limits : list of tuple
            Domain boundaries in each coordinate direction.
            - 2D: [(x0, x1), (y0, y1)]
            - 3D: [(x0, x1), (y0, y1), (z0, z1)]
        delta : float or list of float
            PML thickness in each direction.
        sigma_max : float
            Maximum value of the absorption profile σ.
        p : int
            Polynomial order of the PML profile.
        dim : int
            Problem dimension (2 or 3).
        """

        self.omega = omega
        self.mu = mu
        self.epsilon = epsilon
        self.dim = dim
        
        if dim == 2:
            self.xlim, self.ylim = limits
        elif dim == 3:
            self.xlim, self.ylim, self.zlim = limits
        else:
            raise ValueError("dim must be 2 or 3")
            
        self.delta = delta
        self.sigma_max = sigma_max
        self.pl = pl

    def sigma(self, t, a, b):
        """General sigma function: activated outside a + delta"""

        sigma = bm.zeros_like(t)
        mask_left = (t >= a - self.delta) & (t <= a)
        mask_right = (t >= b) & (t <= b + self.delta)

        sigma[mask_left] = self.sigma_max * (bm.abs(t[mask_left] -a)/self.delta)**self.pl
        sigma[mask_right] = self.sigma_max * (bm.abs(t[mask_right] - b) / self.delta)**self.pl

        return sigma

    def jacobi(self, pp):
        """
        Returns the Jacobian matrix F at each point.
        2D: shape is (N, 2, 2)
        3D: shape is (N, 3, 3)
        """
        x = pp[..., 0]
        y = pp[..., 1]
        N = pp.shape[:-1]
        
        if self.dim == 2:
            shape = N + (2, 2)
            F = bm.zeros(shape, dtype=bm.complex128)
            
            # 获取s1, s2
            s1 = bm.ones_like(x, dtype=bm.complex128)
            s2 = bm.ones_like(y, dtype=bm.complex128)
            
            mask_x = ((x >= self.xlim[0] - self.delta) & (x <= self.xlim[0])) | \
                     ((x >= self.xlim[1]) & (x <= self.xlim[1] + self.delta))
            mask_y = ((y >= self.ylim[0] - self.delta) & (y <= self.ylim[0])) | \
                     ((y >= self.ylim[1]) & (y <= self.ylim[1] + self.delta))
                     
            s1[mask_x] = 1 + 1j * self.sigma(x[mask_x], *self.xlim)
            s2[mask_y] = 1 + 1j * self.sigma(y[mask_y], *self.ylim)
            
            F[..., 0, 0] = s1
            F[..., 1, 1] = s2
            
        else: 
            z = pp[..., 2]
            shape = N + (3, 3)
            F = bm.zeros(shape, dtype=bm.complex128)
            
            s1 = bm.ones_like(x, dtype=bm.complex128)
            s2 = bm.ones_like(y, dtype=bm.complex128)
            s3 = bm.ones_like(z, dtype=bm.complex128)
            
            mask_x = ((x >= self.xlim[0] - self.delta) & (x <= self.xlim[0])) | \
                     ((x >= self.xlim[1]) & (x <= self.xlim[1] + self.delta))
            mask_y = ((y >= self.ylim[0] - self.delta) & (y <= self.ylim[0])) | \
                     ((y >= self.ylim[1]) & (y <= self.ylim[1] + self.delta))
            mask_z = ((z >= self.zlim[0] - self.delta) & (z <= self.zlim[0])) | \
                     ((z >= self.zlim[1]) & (z <= self.zlim[1] + self.delta))
                     
            s1[mask_x] = 1 + 1j * self.sigma(x[mask_x], *self.xlim)
            s2[mask_y] = 1 + 1j * self.sigma(y[mask_y], *self.ylim)
            s3[mask_z] = 1 + 1j * self.sigma(z[mask_z], *self.zlim)
            
            F[..., 0, 0] = s1
            F[..., 1, 1] = s2
            F[..., 2, 2] = s3
            
        return F
    
    def detjacobi(self, pp):
        """Compute the determinant of the Jacobian matrix"""
        F = self.jacobi(pp)
        if self.dim == 2:
            return F[..., 0, 0] * F[..., 1, 1]
        else:
            return F[..., 0, 0] * F[..., 1, 1] * F[..., 2, 2]
    
    @cartesian
    def alpha(self, pp):
        """Construct the medium tensor alpha = mu^{-1} * A"""
        F = self.jacobi(pp)
        detF = self.detjacobi(pp)
        
        FTF = bm.einsum("...ij,...ik->...jk", F, F)
        A = FTF / detF[..., None, None]
        return A / self.mu

    @cartesian
    def beta(self, pp):
        """Construct the medium tensor beta = w^2 * epsilon * A^{-1}"""
        F = self.jacobi(pp)
        detF = self.detjacobi(pp)
        
        FTF = bm.einsum("...ij,...ik->...jk", F, F)
        invA = detF[..., None, None] * bm.linalg.inv(FTF)
        return -(self.omega ** 2) * self.epsilon * invA