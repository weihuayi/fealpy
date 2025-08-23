
from .. import logger
logger.setLevel('WARNING')
from typing import Union
from ..backend import bm
from ..model import PDEModelManager, ComputationalModel
from ..model.curlcurl import CurlCurlPDEDataT

from ..mesh import Mesh
from ..functionspace import FirstNedelecFESpace
from . import ScalarMassIntegrator, CurlCurlIntegrator
from . import VectorSourceIntegrator        
from . import BilinearForm, LinearForm
from ..solver import spsolve
from ..decorator import variantmethod

from ..utils import timer

class CurlCurlUPMLModel(ComputationalModel):
    """
    A model for solving Curl-Curl equations with Robin boundary conditions using the (UPML). 
    Supports both standard and interior penalty FEM.

    Reference:
        https://wnesm678i4.feishu.cn/wiki/VmVowUUuoiUMNSk5LKjcOPGqnWh?from=from_copylink
     """

    def __init__(self, options):
        """
        Initialize the model with configuration options.

        Parameters:
            options (dict): A dictionary containing configuration parameters including:

            - pbar_log (bool): Whether to show progress bar logging.
            - log_level (str): Logging verbosity level (e.g., 'info', 'debug').
            - pde (str or PDEData): Identifier of the PDE problem or a PDE data object.
            - init_mesh (str or Mesh): Mesh type ('uniform_tet', 'uniform_hex', etc.) or a mesh object.
            - nx (int): Number of subdivisions in x-direction for mesh generation.
            - ny (int): Number of subdivisions in y-direction for mesh generation.
            - nz (int): Number of subdivisions in z-direction for mesh generation (3D problems).
            - space_degree (int): Degree of the finite element polynomial space.
            - omega (float): Angular frequency for the wave problem.
            - mu (float): Magnetic permeability (μ) of the medium.
            - epsilon (float): Electric permittivity (ε) of the medium.
            - limits (list of tuples): Computational domain boundaries,
            - delta (float): PML or buffer layer thickness.
            - solver (str or object): Linear solver or solver configuration.
        """
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['nx'], ny=options['ny'], nz=options['nz'])
        self.p = (options['space_degree'])
        self.omega = (options['omega'])
        self.mu = (options['mu'])
        self.epsilon = (options['epsilon'])
        self.limits = (options['limits'])
        self.delta = (options['delta'])
        self.solver = self.options['solver']

    def set_pde(self, pde:Union[CurlCurlPDEDataT, int]  = 1):
        """
        Set the PDE model for the problem.

        Args:
            pde: PDE model instance or string key to get example PDE.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('curlcurl').get_example(pde,**self.options)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tri", **kwargs):
        """
        Initialize the computational mesh.

        Args:
            mesh: Mesh instance or mesh type name as string.
            **kwargs: Additional parameters for mesh generation (e.g., nx, ny).
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh]( **kwargs)
        else:
            self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")


    @variantmethod
    def linear_system(self, mesh, p):
        """
        """
        self.space= FirstNedelecFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")

        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarMassIntegrator(coef=self.pde.beta, q=p+3))
        bform.add_integrator(CurlCurlIntegrator(coef=self.pde.alpha, q=p+3))
        A = bform.assembly()

        lform = LinearForm(self.space)
        lform.add_integrator(VectorSourceIntegrator(self.pde.source, q=p+3))
        F = lform.assembly()
        F = F.astype(bm.complex128)

        return A, F
    
    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import spsolve
        return spsolve(A, F, solver='scipy')
    
    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import minres
        uh, info = minres(A, F)
        return uh

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    @solve.register('cg')
    def solve(self, A, F):
        from ..solver import cg
        uh, info = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(F)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")
        return uh
    
    @variantmethod
    def run(self):
        """
        """
        A, F = self.linear_system(self.mesh, self.p)
        uh = self.space.function()
        uh[:] = self.solve[self.solver](A, F)
    
        return uh