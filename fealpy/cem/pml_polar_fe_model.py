from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian, barycentric

class PMLPolarFEMModel():
    def __init__(self, TD: int, r0 : float, r1 : float, omega : float, mu : float,
                       epsilon : float, s : float = 5.0, p : int = 2, device = None):
        """
        Initialize the parameters of the PDE for the PML

        Parameters:
            TD(int):The dimension of the space
            r0(float): The original regional radius
            r1(float): The radius after expanding the PML
            omega(float): Wavelength
            mu(float): Relative magnetic permeability   
            epsilon(float): Relative permittivity
            s(float): Wavelength scaling factor
            p(int): The power exponent of the stretching function
        """
        self.TD = TD
        self.r0  = r0
        self.r1  = r1
        self.mu  = mu
        self.omega   = omega
        self.epsilon = epsilon
        self.k = omega/3e8
        self.s = s
        self.p = p
        self.kwargs = {"device":device, "dtype":bm.complex128}

    def stretch(self, r):
        """
        Define the stretching function for the PML
        Parameters:
            r(float): The distance from the center of the PML
        Returns:
            The stretching function value at r
        """
        r0 = self.r0
        r1 = self.r1
        mu = self.mu
        omega   = self.omega
        epsilon = self.epsilon
        mu0 = 4*bm.pi*1e-7
        eps0 = 8.854*1e-12

        s = self.s
        p = self.p
        return r + 1j*(s/omega/bm.sqrt(mu0*eps0*mu*epsilon)) * ((r-r0)/(r1-r0))**p

    def dstretch(self, r):
        """
        Define the derivative of the stretching function
        Parameters:
            r(float): The distance from the center of the PML
        Returns:
            The derivative of the stretching function value at r
        """
        r0 = self.r0
        r1 = self.r1
        mu = self.mu
        omega   = self.omega
        epsilon = self.epsilon
        s = self.s
        p = self.p

        mu0 = 4*bm.pi*1e-7
        eps0 = 8.854*1e-12

        c = p*(s/omega/bm.sqrt(mu0*eps0*mu*epsilon))/(r1-r0)**p
        return 1 + 1j*c*(r-r0)**(p-1)

    def jacobi(self, pp):
        """
        The Jacobian matrix of the mapping from the complex plane to the real plane
        parameters:
            pp(tensor): The Cartesian coordinates
        Returns:    
            The Jacobian matrix
        """
        TD = self.TD
        shape = pp.shape + (TD, )
        r    = bm.linalg.norm(pp, axis=-1)
 
        fr   = self.stretch(r)
        dfr  = self.dstretch(r)
        F0 = bm.zeros(shape, **self.kwargs)
        F1 = bm.zeros(shape, **self.kwargs)
        if TD == 2:
            cost = pp[..., 0]/r
            sint = pp[..., 1]/r

            F0 = bm.set_at(F0, (..., 0, 0), dfr*cost)
            F0 = bm.set_at(F0, (..., 0, 1), -fr*sint)
            F0 = bm.set_at(F0, (..., 1, 0), dfr*sint)
            F0 = bm.set_at(F0, (..., 1, 1), fr*cost)

            F1 = bm.set_at(F1, (..., 0, 0), cost)
            F1 = bm.set_at(F1, (..., 0, 1), sint)
            F1 = bm.set_at(F1, (..., 1, 0), -sint/r)
            F1 = bm.set_at(F1, (..., 1, 1), cost/r)
        else:
            cosphi = pp[..., 2]/r
            sinphi = bm.sqrt(1-cosphi**2)
            costhe = pp[..., 0]/(r*sinphi)
            sinthe = pp[..., 1]/(r*sinphi)

            F0 = bm.set_at(F0, (..., 0, 0), dfr*costhe*sinphi)
            F0 = bm.set_at(F0, (..., 0, 1), -fr*sinthe)
            F0 = bm.set_at(F0, (..., 0, 2), fr*costhe*cosphi)
            F0 = bm.set_at(F0, (..., 1, 0), dfr*sinthe*sinphi)
            F0 = bm.set_at(F0, (..., 1, 1), fr*costhe)
            F0 = bm.set_at(F0, (..., 1, 2), fr*sinthe*cosphi)
            F0 = bm.set_at(F0, (..., 2, 0), dfr*cosphi)
            F0 = bm.set_at(F0, (..., 2, 1), 0)
            F0 = bm.set_at(F0, (..., 2, 2), -fr*sinphi)

            F1 = bm.set_at(F1, (..., 0, 0), costhe*sinphi)
            F1 = bm.set_at(F1, (..., 0, 1), sinthe*sinphi)  
            F1 = bm.set_at(F1, (..., 0, 2), cosphi)
            F1 = bm.set_at(F1, (..., 1, 0), -sinthe/r)
            F1 = bm.set_at(F1, (..., 1, 1), costhe/r)
            F1 = bm.set_at(F1, (..., 1, 2), 0)
            F1 = bm.set_at(F1, (..., 2, 0), costhe*cosphi/r)
            F1 = bm.set_at(F1, (..., 2, 1), sinthe*cosphi/r)
            F1 = bm.set_at(F1, (..., 2, 2), -sinphi/r)
        
        F = bm.einsum("...ij, ...jk->...ik", F0, F1)
        return F

    def detjacobi(self, pp):
        """
        The Jacobian determinant of the mapping from the complex plane to the real plane
        parameters:
            pp(tensor): The Cartesian coordinates
        Returns:    
            The Jacobian determinant
        """ 
        TD = self.TD
        r    = bm.linalg.norm(pp, axis=-1)
        fr   = self.stretch(r)
        dfr  = self.dstretch(r)
        detF = fr**(TD-1) * dfr/r**(TD-1)
        return detF

    @cartesian
    def alpha(self, pp):
        """
        The PML alpha parameter
        parameters:
            pp(tensor): The Cartesian coordinates
        Returns:    
            The alpha parameter
        """
        TD = self.TD
        r0 = self.r0
        mu = self.mu

        r       = bm.linalg.norm(pp, axis=-1)
        isInPML = r > r0
        F       = self.jacobi(pp[isInPML])
        detF    = self.detjacobi(pp[isInPML])
        if TD == 2:
            val = bm.zeros(pp.shape[:-1], **self.kwargs)
            val = bm.set_at(val,(slice(None)), 1/mu)

            val[isInPML] = val[isInPML]/detF
        else:
            val = bm.zeros(pp.shape + (3, ), **self.kwargs)
            val = bm.set_at(val, (..., 0, 0), 1/mu)
            val = bm.set_at(val, (..., 1, 1), 1/mu)
            val = bm.set_at(val, (..., 2, 2), 1/mu)

            FTF = bm.einsum("...ij, ...ik->...jk", F, F)
            val[isInPML] = FTF / mu / detF[:, None, None]
        return val

    @cartesian
    def beta(self, pp):
        """
        The PML beta parameter
        parameters:
            pp(tensor): The Cartesian coordinates
        Returns:    
            The beta parameter
        """
        TD = self.TD
        r0 = self.r0
        epsilon = self.epsilon
        k = self.k

        r       = bm.linalg.norm(pp, axis=-1)
        isInPML = r > r0
        F       = self.jacobi(pp[isInPML])
        detF    = self.detjacobi(pp[isInPML])
        val = bm.zeros(pp.shape + (TD, ), **self.kwargs)

        if TD == 2:
            val = bm.set_at(val, (..., 0, 0), k**2 * epsilon)
            val = bm.set_at(val, (..., 1, 1), k**2 * epsilon)
            val[isInPML] = k**2 * epsilon * detF[:, None, None]
        else:
            val = bm.set_at(val, (..., 0, 0), k**2 * epsilon)
            val = bm.set_at(val, (..., 1, 1), k**2 * epsilon)
            val = bm.set_at(val, (..., 2, 2), k**2 * epsilon)

            FTF    = bm.einsum("...ij, ...ik->...jk", F, F)
            invFTF = bm.linalg.inv(FTF)
            val[isInPML] = k**2 * epsilon * detF[:, None, None] * invFTF
        return val