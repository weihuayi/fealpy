import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags, bmat

from fealpy.decorator import barycentric
from ..functionspace import LagrangeFESpace
from ..fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from ..fem import VectorDiffusionIntegrator
from ..fem import ScalarMassIntegrator, ScalarConvectionIntegrator
from ..fem import VectorViscousWorkIntegrator, PressWorkIntegrator
from ..fem import BilinearForm, MixedBilinearForm
from ..fem import LinearForm
from ..fem import VectorSourceIntegrator, ScalarSourceIntegrator
from ..fem import VectorConvectionIntegrator


class NSFEMSolver:
    def __init__(self, mesh, dt, uspace, pspace, rho=1.0, mu=1.0, q=4):
        #self.model = model
        self.mesh = mesh
        self.uspace = uspace 
        self.pspace = pspace
        self.rho = rho
        self.mu = mu
        self.q = q
        self.dt = dt

        ##\rho u
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=q))
        self.M = bform.assembly() 
        
        ##mu * \laplace u
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorDiffusionIntegrator(c=self.mu, q=q))
        self.S = bform.assembly() 
        
        ##\laplace p
        bform = BilinearForm(self.pspace)
        bform.add_domain_integrator(ScalarDiffusionIntegrator(q=q))
        self.SP = bform.assembly() 
        
        ##\nabla p
        bform = MixedBilinearForm((self.pspace,), 2*(self.uspace,)) 
        bform.add_domain_integrator(PressWorkIntegrator(q=q)) 
        self.AP = bform.assembly()
    
    #u \cdot u   \approx   u^n \cdot u^{n+1}
    def ossen_A(self,u0):
        M = self.M
        AP = self.AP
        rho = self.rho
        S = self.S
        SP = self.SP
        dt = self.dt
 
        @barycentric
        def coef(bcs, index):
            if callable(rho):
                return rho(bcs,index)*u0(bcs,index)
            else:
                return rho*u0(bcs,index)
            
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorConvectionIntegrator(c=coef, q=self.q))
        C = bform.assembly() 

        A0 = 1/dt*M+S+C
        A = bmat([[1/dt*M+S+C,  -AP],\
                [AP.T, None]], format='csr')
        return A

    def Ossen_b(self, un): 
        dt = self.dt
        pgdof = self.pspace.number_of_global_dofs()
        M = self.M
        
        b = 1/dt * M@un.flatten()
        b = np.hstack((b,[0]*pgdof))
        return b
    
    def slip_stick_boundary(self, stick_area=None):
        pass

    def output(self, phi, u, timestep, output_dir=',/', filename_prefix='test'):
        mesh = self.space.mesh
        if output_dir != 'None':
            mesh.nodedata['phi'] = phi
            mesh.nodedata['velocity'] = u
            fname = os.path.join(output_dir, f'{filename_prefix}_{timestep:010}.vtu')
            mesh.to_vtk(fname=fname)
