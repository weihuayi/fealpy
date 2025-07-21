
from .. import logger
logger.setLevel('WARNING')
from typing import Union
from ..backend import bm
from ..model import PDEModelManager, ComputationalModel
from ..model.curlcurl import CurlCurlPDEDataT

from ..mesh import Mesh
from ..functionspace import FirstNedelecFESpace
from . import ScalarMassIntegrator, CurlCurlIntegrator, BoundaryFaceMassIntegrator
from . import CurlJumpPenaltyIntergrator 
from . import BoundaryFaceSourceIntegrator, VectorSourceIntegrator        
from . import BilinearForm, LinearForm
from ..solver import spsolve
from ..decorator import variantmethod

from ..utils import timer
import matplotlib.pyplot as plt


class CurlCurlLFEMModel(ComputationalModel):
    """
    A model for solving Curl-Curl equations with Robin boundary conditions using the (LFEM). 
    Supports both standard and interior penalty FEM.

    Reference:
        https://wnesm678i4.feishu.cn/wiki/ZBGuwQMe4iumYikfmyAc3Snence
     """

    def __init__(self, options):
        """
        Initialize the model with configuration options.

        Parameters:
            options (dict): A dictionary containing configuration parameters including:
                - pbar_log (bool): Whether to show progress bar logging.
                - log_level (str): Logging verbosity level.
                - pde (str or PDEData): PDE problem identifier or data.
                - init_mesh (str or Mesh): Mesh type or a mesh object.
                - nx (int): Number of subdivisions in x-direction for mesh.
                - ny (int): Number of subdivisions in y-direction for mesh.
                - space_degree (int): Degree of finite element polynomial space.
                - wave_number (float): Wave number parameter for PDE.
                - gamma (float): Penalty parameter for interior penalty FEM.
        """
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['nx'], ny=options['ny'])
        self.set_space_degree(options['space_degree'])
        self.set_wave_number(options['wave_number'])
        self.set_gamma(options['gamma'])
        self.solver = self.options['solver']
        self.method = self.options['method']

    def set_pde(self, pde:Union[CurlCurlPDEDataT, int]  = 1):
        """
        Set the PDE model for the problem.

        Args:
            pde: PDE model instance or string key to get example PDE.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('curlcurl').get_example(pde)
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
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")


    def set_space_degree(self, p: int = 1) :
        """
        Set the polynomial degree of the finite element space.

        Args:
            p: The polynomial degree.
        """
        self.p = p
        
    def set_wave_number(self, k: int):
        """
        Set the wave number for the Helmholtz equation.

        Args:
            k: Wave number.
        """
        self.k = k

    def set_gamma(self, gamma: float):
        """
        Set the penalty parameter gamma used in interior penalty FEM.

        Args:
            gamma: Penalty parameter.
        """
        self.gamma = gamma

    @variantmethod("standard")
    def linear_system(self, mesh, p):
        """
        """
        self.pde.set(k=self.k)
        self.space= FirstNedelecFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")

        D = CurlCurlIntegrator(coef=1, q=p+3)
        M = ScalarMassIntegrator(coef=-self.k**2, q=self.p+2)
        R = BoundaryFaceMassIntegrator(coef=-1j, q=p+3)
 
        beform = BilinearForm(self.space)
        beform.add_integrator(D)
        beform.add_integrator(M)
        beform.add_integrator(R)
        A = beform.assembly() 

        f = VectorSourceIntegrator(self.pde.source, q=self.p+2)
        Vr = BoundaryFaceSourceIntegrator(self.pde.robin, q=p+3)

        leform = LinearForm(self.space)
        leform.add_integrator(f)
        leform.add_integrator(Vr)
        F = leform.assembly()

        return A, F
    
    @linear_system.register('penalty')
    def linear_system(self, mesh, p):
        """
        """
        self.pde.set(k=self.k)
        self.space= FirstNedelecFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")
        
        D = CurlCurlIntegrator(coef=1, q=p+3)
        M = ScalarMassIntegrator(coef=-self.k**2, q=self.p+2)
        R = BoundaryFaceMassIntegrator(coef=-1j, q=p+3)
        G = CurlJumpPenaltyIntergrator(coef=self.gamma, q=self.p+2)
 
        beform = BilinearForm(self.space)
        beform.add_integrator(D)
        beform.add_integrator(M)
        beform.add_integrator(R)
        beform.add_integrator(G)
        A = beform.assembly() 

        f = VectorSourceIntegrator(self.pde.source, q=self.p+2)
        Vr = BoundaryFaceSourceIntegrator(self.pde.robin, q=p+3)

        leform = LinearForm(self.space)
        leform.add_integrator(f)
        leform.add_integrator(Vr)
        F = leform.assembly()

        return A, F
    
    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import spsolve
        return spsolve(A, F, solver='scipy')

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
        self.pde.set(k=self.k)
        A, F = self.linear_system[f"{self.method}"](self.mesh, self.p)
        uh = self.space.function(dtype=bm.complex128)
        uh[:] = self.solve[self.solver](A, F)
        error = self.mesh.error(self.pde.solution, uh.value)

        return uh, error
