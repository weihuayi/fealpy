from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

from fealpy.model.curlcurl.curlcurl_sphere_pml import CurlCurlSpherePML

class Exp0003(CurlCurlSpherePML):
    """
    Dipole antenna example in a spherical domain:

        curl(1/mu_r * curl(E)) - k^2 * epsilon_r * E = J       in Ω
        n × E = g                                             on Γ_D
        n × (1/mu_r * curl(E)) - Y * n × (n × E) = f         on Γ_R

    Parameters:
        mu_r = 1, epsilon_r = 1
        Y = 1/Z, where 
        Z = sqrt((mu_0 * mu_r) / (epsilon_0 * epsilon_r - i * sigma / omega))
        sigma = 6e7 S/m

    This models the excitation of a dipole at the center of the spherical domain,
    with a Dirichlet condition applied on a small patch representing the feed.
    """

    def __init__(self, r0 = 1.9, r1 = 2.4, mu=1, epsilon=1, s = 5.0, p = 2):
        """
        Initialize dipole antenna problem.

        Parameters
        ----------
        r0 : float
            Inner radius (m), default 1.9.
        r1 : float
            Outer radius (m), default 2.4.
        mu : float
            Magnetic permeability (H/m), default 1.
        epsilon : float
            Relative permittivity, default 1.
        s : float
            PML thickness (m).
        p : int
            Order of Legendre polynomials for the PML.
        """
        self.mu0   = 4*bm.pi*1e-7
        self.eps0  = 8.854e-12
        self.sigma = 6e7
        self.f     = 0.07498
        self.omega = 2*bm.pi*self.f*1e9
        self.k = self.omega / 3e8
        self.Z     = bm.sqrt(self.mu0/(self.eps0-1j*self.sigma/self.omega))
        self.Y     = 1j*self.omega*self.mu0/self.Z

        super(Exp0003, self).__init__(r0, r1, self.omega, mu, epsilon, s, p)

    @cartesian
    def source(self, p):
        shape = p.shape[:-1]
        return bm.zeros((*shape, 3), dtype=bm.complex128)

    @cartesian
    def dirichlet(self, p, n):
        """
        Dirichlet boundary condition for the dipole feed.

        Parameters
        ----------
        p : array_like
            Cartesian coordinates of points.
        n : array_like
            Normal vectors at boundary points.

        Returns
        -------
        val : array_like
            Tangential component of the Dirichlet field (n × E) at boundary points.
        """
        r = bm.linalg.norm(p, axis=-1)
        flag = (r < 0.1) & (bm.abs(p[..., 2]) < 0.0050001)
        val = bm.zeros_like(p)
        val[flag, 2] = 10
        return bm.cross(n, val)
