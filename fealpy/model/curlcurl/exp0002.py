from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher3d
from fealpy.model.curlcurl.curlcurl_upml import CurlCurlUPML

class Exp0002(CurlCurlUPML, BoxMesher3d):
    """
    3D Maxwell's curl-curl problem with UPML (Uniaxial Perfectly Matched Layer).
    
    Governing equation (frequency domain Maxwell's equations with source term f):
    
        curl(curl(u)) - ω^2 ε u = f    in Ω ⊂ R^3
        u = 0                         on ∂Ω  (Dirichlet boundary condition)
    
    where:
        - u(x) ∈ C^3 is the vector field solution (e.g., electric or magnetic field)
        - ω is the angular frequency
        - ε is the permittivity (absorbed into model coefficients)
        - f is the vector source term
        - UPML layers are included to absorb outgoing waves
    
    Source:
        A Gaussian-like source centered at (0, 0, 0.5) with frequency scaling.
        Each component of f(x) is identical, i.e. isotropic excitation.
    
    Boundary condition:
        Homogeneous Dirichlet boundary condition: u = 0 on ∂Ω.
    """
    
    def __init__(self, options: dict = {}):
        """
        Initialize the problem instance by calling the parent class CurlCurlUPML.
        """
        BoxMesher3d.__init__(self, options.pop("box"))
        CurlCurlUPML.__init__(self, **options)
        super().__init__(**options) 


    @cartesian
    def source(self, pp):
        """
        Define the source term f(x) = (g, g, g).
        
        Args:
            pp (ndarray): Evaluation points of shape (..., 3).
        
        Returns:
            ndarray: Vector-valued source of shape (..., 3).
        
        The Gaussian source is defined as:
            g(x) = exp( prefactor * (x1^2 + x2^2 + (x3 - 0.5)^2) )
        where
            prefactor = (-4 * ω / π)^3.
        """
        omega = self.omega
        x1, x2, x3 = pp[..., 0], pp[..., 1], pp[..., 2]

        prefactor = (-4 * omega / bm.pi) ** 3
        g = bm.exp(prefactor * (x1 ** 2 + x2 ** 2 + (x3 - 0.5) ** 2))

        f = bm.stack([g, g, g], axis=-1)  
        return f


    @cartesian
    def dirichlet(self, pp, n):
        """
        Homogeneous Dirichlet boundary condition.
        
        Args:
            pp (ndarray): Boundary evaluation points of shape (..., 3).
            n (ndarray): Outward unit normals of shape (..., 3).
        
        Returns:
            ndarray: Zero vector of shape (..., 3).
        
        This enforces u = 0 on ∂Ω.
        """
        return bm.zeros(pp.shape, dtype=bm.complex128)
