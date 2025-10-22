from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm

class HelmholtzUpml:
    """
    Unified handling of the Helmholtz equation in 2D and 3D PML layers:

        ∇·(A ∇ũ) + k² J ũ = f  in Ω
        ũ = g on Γ

    where:
        A = J * B^{-T} * B^{-1}
        J = det(B)
        B is the Jacobian matrix of the complex coordinate stretching.

    In PML layers:
        - For 2D: B = diag(s1(x), s2(y))
        - For 3D: B = diag(s1(x), s2(y), s3(z))
        - Each stretching factor is defined as:
              s_j = 1 + i·σ_j(x_j)/ω
    """

    
    def __init__(self, k, limits, delta, 
                 sigma_max=10.0, p=2, dim=2):
        """
      Initialize PML parameters.

        Parameters:
        omega : float
            Angular frequency of the wave.
        k : float
            Wave number.
        domain_limits : list of tuples
            Domain boundaries in each coordinate direction, e.g., [(a0, a1), (b0, b1), ...].
        delta : float
            Thickness of the PML layer.
        sigma_max : float
            Maximum value of the absorption coefficient σ.
        p : int
            Polynomial order of the PML profile.
        dim : int
            Problem dimension (2 or 3).

        """
        self.dim = dim
        self.delta = delta
        self.sigma_max = sigma_max
        self.p = p
        self.k = k
        
        # 设置各方向限制
        if dim == 2:
            self.xlim, self.ylim = limits
        elif dim == 3:
            self.xlim, self.ylim, self.zlim = limits
        else:
            raise ValueError("dim must be 2 or 3")
    
    def sigma(self, t, a, b):
        """Sigma function: activated near the boundaries"""
        sigma = bm.zeros_like(t)
        mask_left = (t >= a - self.delta) & (t <= a)
        mask_right = (t >= b) & (t <= b + self.delta)
        
        sigma[mask_left] = self.sigma_max * (bm.abs(t[mask_left] -a)/self.delta)**self.p
        sigma[mask_right] = self.sigma_max * (bm.abs(t[mask_right] - b)/self.delta)**self.p
        
        return sigma
                  
    def factors(self, pp):
        """Calculate stretching factors s_j = 1 + i·σ_j/ω"""  
        x = pp[..., 0]
        y = pp[..., 1]
        

        s1 = bm.ones_like(x, dtype=bm.complex128)
        s2 = bm.ones_like(y, dtype=bm.complex128)
        
        mask_x = ((x >= self.xlim[0] - self.delta) & (x <= self.xlim[0])) | \
                 ((x >= self.xlim[1]) & (x <= self.xlim[1] + self.delta))
        s1[mask_x] = 1 + 1j * self.sigma(x[mask_x], *self.xlim) 
        

        mask_y = ((y >= self.ylim[0] - self.delta) & (y <= self.ylim[0])) | \
                 ((y >= self.ylim[1]) & (y <= self.ylim[1] + self.delta))
        s2[mask_y] = 1 + 1j * self.sigma(y[mask_y], *self.ylim)
        
        if self.dim == 2:
            return s1, s2
        else:
            z = pp[..., 2]
            s3 = bm.ones_like(z, dtype=bm.complex128)
            mask_z = ((z >= self.zlim[0] - self.delta) & (z <= self.zlim[0])) | \
                     ((z >= self.zlim[1]) & (z <= self.zlim[1] + self.delta))
            s3[mask_z] = 1 + 1j * self.sigma(z[mask_z], *self.zlim)
            return s1, s2, s3
    
    def jacobi(self, pp):
        """Construct the Jacobi matrix B = diag(s1, s2, ...)"""
        s_factors = self.factors(pp)
        N = pp.shape[:-1]
        
        if self.dim == 2:
            s1, s2 = s_factors
            B = bm.zeros(N + (2, 2), dtype=bm.complex128)
            B[..., 0, 0] = s1
            B[..., 1, 1] = s2
        else:
            s1, s2, s3 = s_factors
            B = bm.zeros(N + (3, 3), dtype=bm.complex128)
            B[..., 0, 0] = s1
            B[..., 1, 1] = s2
            B[..., 2, 2] = s3
            
        return B
    
    def detjacobi(self, pp):
        B = self.jacobi(pp)
        if self.dim == 2:
            return B[..., 0, 0] * B[..., 1, 1]
        else:
            return B[..., 0, 0] * B[..., 1, 1] * B[..., 2, 2]
    
    @cartesian
    def alpha(self, pp):
        """Compute A = J · B^{-T} · B^{-1}"""

        B = self.jacobi(pp)
        J = self.detjacobi(pp)
        inv_B = bm.linalg.inv(B)
        inv_BT_inv_B = bm.zeros_like(B)

        if self.dim == 2:
            inv_BT_inv_B[..., 0, 0] = inv_B[..., 0, 0]**2
            inv_BT_inv_B[..., 1, 1] = inv_B[..., 1, 1]**2
        else:
            inv_BT_inv_B[..., 0, 0] = inv_B[..., 0, 0]**2
            inv_BT_inv_B[..., 1, 1] = inv_B[..., 1, 1]**2
            inv_BT_inv_B[..., 2, 2] = inv_B[..., 2, 2]**2
        
        # A = J * inv_BT_inv_B
        A = J[..., None, None] * inv_BT_inv_B
        return -A
       
    @cartesian
    def beta(self, pp):
        """Reaction coefficient: k² · J"""
        J = self.detjacobi(pp)
        return (self.k**2) * J  