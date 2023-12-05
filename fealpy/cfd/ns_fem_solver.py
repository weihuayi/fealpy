import numpy as np
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm
from ..fem import LinearForm

from ..fem import ScalarConvectionIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ScalarMassIntegrator

class NSFEMSolver:
    def __init__(self, model, mesh, p=(2, 1), rho=1.0, mu=1.0, q=5):
        self.model = model
        self.mesh = mesh
        self.uspace = LagrangeFESpace(mesh, p=p[0])
        self.pspace = LagrangeFESpace(mesh, p=p[1])
        
        ##rho(u ,v)
        bform = BilinearForm(space)
        bform.add_domain_integrator(ScalarMassIntegrator())
        self.M = bform.assembly() 

    def output(self, phi, u, timestep, output_dir=',/', filename_prefix='test'):
        mesh = self.space.mesh
        if output_dir != 'None':
            mesh.nodedata['phi'] = phi
            mesh.nodedata['velocity'] = u
            fname = os.path.join(output_dir, f'{filename_prefix}_{timestep:010}.vtu')
            mesh.to_vtk(fname=fname)
