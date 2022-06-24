import numpy as np
from scipy.special import jv
from ..decorator import cartesian

class HelmholtzData3d():

    def __init__(self, k=100):
        """
        @brief 

        @param[in] k wave number
        """
        self.k = k

    def domain(self):
        return [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]

    @cartesian
    def solution(self, p):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)

        val = np.zeros(x.shape, dtype=np.complex128)
        val[:] = np.cos(k*r)/k
        c = complex(np.cos(k), np.sin(k))/complex(jv(0, k), jv(1, k))/k
        val -= c*jv(0, k*r)
        return val

    @cartesian
    def gradient(self, p):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        val = np.zeros(p.shape, dtype=np.complex128)
        t0 = k*np.sin(k*r)/(4*r)
        c = complex(np.cos(k), np.sin(k))/complex(jv(0, k), jv(1, k))
        t1 = c*jv(1, k*r)/r
        t2 = t1 - t0
        val[..., 0] = t2*x
        val[..., 1] = t2*y
        val[..., 2] = t2*y
        return val

    @cartesian
    def source(self, p):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        val = complex(np.sin(k*r)/r, 0.0)
        return val

    @cartesian
    def robin(self, p, n):
        k = self.k
        grad = self.gradient(p) # (NQ, NE, 3)
        val = np.sum(grad*n, axis=-1)
        kappa = np.complex(0.0, k) 
        val += kappa*self.solution(p) 
        return val, kappa


    def symbolic_com(self):
        import sympy as sp
        x, y, z, k = sp.symbols('x, y, z, k', real=True)
        r = sp.sqrt(x**2 + y**2 + z**2)
        J0k = sp.besselj(0, k)
        J1k = sp.besselj(1, k)
        J0kr = sp.besselj(0, k*r)
        u = sp.cos(k*r)/4 - J0kr*(sp.cos(k) + sp.I*sp.sin(k))/(J0k + sp.I*J1k)/k
        print(u.diff(x))
        print(u.diff(y))
        print(u.diff(z))
