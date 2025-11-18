from fealpy.backend import bm
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher2d

class RisingBubblePDE(BoxMesher2d):
    def __init__(self,domain, d, area, rho1, rho2, 
                      mu1, mu2, eta, gamma, lam):
        """
        Parameters
            domain : list
                Computational domain, [x0,x1,y0,y1].
            d : float
                Diameter of the area.
            area : float
            rho1 : float
                Density of fluid 1.
            rho2 : float
                Density of fluid 2.
            mu1 : float
                Viscosity of fluid 1.
            mu2 : float
                Viscosity of fluid 2.
            eta : float
                Thickness of the interface.
            gamma : float
                Mobility of the interface.
            lam : float
                Surface tension coefficient.
        """
        self.box = domain
        self.d = d
        self.area = area
        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.g = 9.8
        
        self.dimensionless()
        
    def dimensionless(self):
        ref_length = self.d
        ref_velocity = (self.g*self.d)**0.5
        ref_rho = min(self.rho1,self.rho2)
        ref_mu = ref_rho*ref_length*ref_velocity
        
        self.area /= ref_length**2
        self.box /= ref_length
        self.eta /= ref_length
        
        self.rho1 /= ref_rho
        self.rho2 /= ref_rho
        self.mu1 /= ref_mu
        self.mu2 /= ref_mu
        
        self.g = 1.0
        self.d /= ref_length
        super().__init__(box=self.box)
    
    @cartesian
    def init_phase(self , p):
        """
        Initial phase function.
        """
        x = p[:,0]
        y = p[:,1]
        r = bm.sqrt(x**2 + y**2)
        val = -bm.tanh(r - 0.5*self.d)/ (self.eta)
        return val
    
    @cartesian
    def phase_forse(self, p, t):
        """
        Phase function source term.
        """
        return bm.zeros(p.shape[0], dtype=bm.float64)
    
    @cartesian
    def init_velocity(self, p):
        """
        Initial velocity.
        """
        val = bm.zeros(p.shape, dtype=bm.float64)
        return val
    
    @cartesian
    def velocity_forse(self, p, t):
        """
        Velocity source term.
        """
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[...,1] = -self.g
        return val
    
    @cartesian
    def velocity_dirichlet_bc(self, p, t):
        """
        Velocity Dirichlet boundary condition.
        """
        val = bm.zeros(p.shape, dtype=bm.float64)
        return val
    
    @cartesian
    def init_pressure(self, p):
        """
        Initial pressure.
        """
        val = bm.zeros(p.shape[0], dtype=bm.float64)
        return val
    