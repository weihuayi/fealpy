import matplotlib.pyplot as plt
from typing import Union

from ..backend import bm
from ..mesh import Mesh, TriangleMesh
from ..functionspace import TensorFunctionSpace, LagrangeFESpace,RaviartThomasFESpace
from ..fem import BilinearForm, LinearForm, BlockForm
from ..fem import ScalarSourceIntegrator, ScalarNeumannBCIntegrator, ScalarMassIntegrator, GradPressureIntegrator, DivIntegrator
from ..model import PDEModelManager, ComputationalModel
from ..decorator import variantmethod
from ..solver import  DarcyForchheimerTPDv
from ..tools.show import show_error_table, showmultirate


class DarcyForchheimerFEMModel(ComputationalModel):

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get('pbar_log', False),
                         log_level=options.get('log_level', 'INFO'))
        self.set_pde(options.get('pde', 3))
        self.set_init_mesh(options.get('init_mesh', "uniform_tri"),
                           nx=options.get('nx',10), ny=options.get('ny', 10))
        self.set_space_degree(options.get('pdegree', 1), options.get('udegree', 0))

    def set_pde(self, pde: Union[int, object] = 1):
        """Accept PDE id or PDE instance (compatible with PDEModelManager)."""
        if isinstance(pde, int) or isinstance(pde, str):
            self.pde = PDEModelManager('darcyforchheimer').get_example(pde)
            self.logger.info(f"PDE initialized from id: '{pde}'")
        else:
            self.pde = pde
            self.logger.info(f"PDE initialized from instance: {type(pde).__name__}")

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", **kwargs):
        """Create or set mesh. If mesh is str, use PDE's init_mesh factory."""
        if isinstance(mesh, str):
            # assume self.pde.init_mesh exists (like in your example)
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
            self.logger.info(f"Mesh type: '{mesh}' created with {kwargs}")
        else:
            self.mesh = mesh
            self.logger.info(f"Custom mesh provided: {type(mesh).__name__}")

        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh: nodes={NN}, cells={NC}")

    def set_space_degree(self, pdegree: int = 1, udegree: int = 0):
        """Set FE spaces: pressure P^pdegree (continuous), velocity P^{udegree} (discontinuous)"""
        self.pdegree = 1
        self.udegree = 1

        self.pspace = LagrangeFESpace(self.mesh, p=self.pdegree, ctype='D')
        self.uspace = RaviartThomasFESpace(self.mesh, p=self.udegree)
        # space = LagrangeFESpace(self.mesh, p=self.udegree, ctype='D')
        # self.uspace = TensorFunctionSpace(space, (-1,2))

        # FE functions

        self.u0 = self.uspace.function() + 1 # used for Mu.coef callback
        self.p0 = self.pspace.function() + 1

        # build forms but do not assemble M (it will be assembled inside solvers)
        self.u_bform = BilinearForm(self.uspace)
        # linear viscous mass integrator (mu part)
        self.u_bform.add_integrator(ScalarMassIntegrator(coef=self.pde.mu, q=4))
        # Mu is the integrator for beta*|u| part; set its coef dynamically in solvers
        self.Mu = ScalarMassIntegrator(q=4)
        self.u_bform.add_integrator(self.Mu)

        # pressure-velocity coupling
        self.p_bform = BilinearForm(self.pspace, self.uspace)
        self.p_bform.add_integrator(DivIntegrator(coef=-1, q=3))

        # RHS forms (dont assemble here)
        self.ulform = LinearForm(self.uspace)
        self.ulform.add_integrator(ScalarSourceIntegrator(self.pde.f, q=4))
        self.plform = LinearForm(self.pspace)
        self.plform.add_integrator(ScalarSourceIntegrator(self.pde.g, q=4))
         
        # uspace, pspace = self.space()
        # ugdof = uspace.number_of_global_dofs()
        # G_apply = pspace.set_neumann_bc(gd)
        # F = bm.zeros(ugdof, dtype=bm.float64)
        # b_apply = bm.concatenate([G_apply,F],axis=0)
        # b = b - b_apply
        # # try to add neumann if exists
        # try:
        #     self.plform.add_integrator(ScalarNeumannBCIntegrator(source=self.pde.neumann, q=4))
        # except Exception:
        #     pass

    def linear_system(self):

        B = self.p_bform.assembly().T
        f = self.ulform.assembly()
        g = self.plform.assembly()
        uspace = self.uspace
        G = uspace.set_neumann_bc(self.pde.pressure)
        f = f - G
        
        return B, f, g

    def apply_bc(self, A, F):
     
        return A, F


    @variantmethod("TPDv")
    def solve(self, maxIt: int = 1000, tol: float = 1e-6,
              gamma0: float = 10, stepsize: float = 0.4, scaleu: float = 0.8):


        B, f, g = self.linear_system()
        # initial u0/p0: if not provided, use current stored ones or random
        if bm.any(self.u0[:] == 0):
            self.u0[:] = bm.random.rand(self.uspace.number_of_global_dofs())
        if bm.any(self.p0[:] == 0):
            self.p0[:] = bm.random.rand(self.pspace.number_of_global_dofs())
        uh = self.uspace.function()
        ph = self.pspace.function()
        # call your TPDv (expects B,f,g,u_bform,Mu,pde,u0,p0,...)
        uh[:], ph[:], resu, resp = DarcyForchheimerTPDv(B, f, g, self.u_bform, self.Mu,
                                  self.pde, self.u0, self.p0,
                                  maxIt=maxIt, tol=tol, gamma0=gamma0,
                                  stepsize=stepsize, scaleu=scaleu).TPDv()
        ip1 = self.mesh.integral(ph)/(sum(self.mesh.entity_measure('cell')))
        #ph[:] = ph[:] - ip1 + pspace.integralalg.integral(pde.pressure)/(sum(mesh.entity_measure('cell')))
        # ph[:] = ph[:] -  ip1 
        eu = self.mesh.error(self.pde.velocity, uh)
        ep = self.mesh.error(self.pde.pressure, ph)
        
        self.TPD_u = uh[:].copy()

        print("eu:", eu)
        print("ep:", ep)
        
        # print(bm.max(self.direct_u - self.TPD_u))


        return uh, ph, {'residual': resu, 'pressure_residual': resp}, (eu, ep)
    
