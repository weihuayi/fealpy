import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags, bmat

from ..functionspace import LagrangeFESpace
from ..fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from ..fem import ScalarMassIntegrator, ScalarConvectionIntegrator
from ..fem import VectorViscousWorkIntegrator, PressWorkIntegrator
from ..fem import BilinearForm, MixedBilinearForm
from ..fem import LinearForm
from ..fem import VectorSourceIntegrator, ScalarSourceIntegrator
from ..fem import VectorConvectionIntegrator

class NSFEMSolver:
    def __init__(self, mesh, p=(2, 1), rho=1.0, mu=1.0, q=5):
        #self.model = model
        self.mesh = mesh
        self.uspace = LagrangeFESpace(mesh, p=p[0])
        self.pspace = LagrangeFESpace(mesh, p=p[1])
        
        ##rho(u ,v)
        bform = BilinearForm(self.uspace)
        bform.add_domain_integrator(ScalarMassIntegrator())
        self.M = bform.assembly() 
        
        ##(\nabla u, \nabla v)
        bform = BilinearForm(self.uspace)
        bform.add_domain_integrator(ScalarDiffusionIntegrator())
        self.S = bform.assembly() 
            
        bform = MixedBilinearForm((pspace,), 2*(uspace,)) 
        bform.add_domain_integrator(PressWorkIntegrator()) 
        self.AP = bform.assembly()
    
    def ossen_A(self,u0):
        M = self.M
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorConvectionIntegrator(c=u0))
        C = bform.assembly() 

        A0 = bmat([[AU+S+AUC, None],[None, AU+S+AUC]], format='csr')  
        
        A = bmat([[A0,  -AP, None],\
                [AP.T, 1e-8*ASP, None],\
                [None,None,AT+ATU]], format='csr')
        return M





    def output(self, phi, u, timestep, output_dir=',/', filename_prefix='test'):
        mesh = self.space.mesh
        if output_dir != 'None':
            mesh.nodedata['phi'] = phi
            mesh.nodedata['velocity'] = u
            fname = os.path.join(output_dir, f'{filename_prefix}_{timestep:010}.vtu')
            mesh.to_vtk(fname=fname)
