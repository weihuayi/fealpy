
from .. import logger
logger.setLevel('WARNING')
from typing import Union
from ..backend import bm
from ..model import PDEModelManager, ComputationalModel
from ..model.helmholtz import HelmholtzPDEDataT

from ..mesh import Mesh
from ..functionspace import LagrangeFESpace
from . import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarRobinBCIntegrator 
from . import JumpPenaltyIntergrator 
from . import ScalarSourceIntegrator, ScalarRobinSourceIntegrator         
from . import BilinearForm, LinearForm
from ..solver import spsolve
from ..decorator import variantmethod

from ..utils import timer
import matplotlib.pyplot as plt


class HelmholtzLFEMModel(ComputationalModel):
    """
    A model for solving Helmholtz equations with Robin boundary conditions using the (LFEM). 
    Supports both standard and interior penalty FEM.

    Reference:
        https://wnesm678i4.feishu.cn/wiki/XjoswuhmniwRNgk7gXlcf0SRn6f?from=from_copylink
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
        self.k = self.options['wave_number']
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['nx'], ny=options['ny'])
        self.set_space_degree(options['space_degree'])
        self.set_gamma(options['gamma'])
        self.solver = self.options['solver']
        self.method = self.options['method']

    def set_pde(self, pde:Union[HelmholtzPDEDataT, int]  = 1):
        """
        Set the PDE model for the problem.

        Args:
            pde: PDE model instance or string key to get example PDE.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('helmholtz').get_example(pde, k = self.k)
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
        self.space= LagrangeFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")

        D = ScalarDiffusionIntegrator(coef=1, q=self.p+2)
        M = ScalarMassIntegrator(coef=-self.k**2, q=self.p+2)
        R = ScalarRobinBCIntegrator(coef=self.k * 1j, q=self.p+2)
 
        beform = BilinearForm(self.space)
        beform.add_integrator(D)
        beform.add_integrator(M)
        beform.add_integrator(R)
        A = beform.assembly() 

        f = ScalarSourceIntegrator(self.pde.source, q=self.p+2)
        Vr = ScalarRobinSourceIntegrator(self.pde.robin, q=self.p+2)

        leform = LinearForm(self.space)
        leform.add_integrator(f)
        leform.add_integrator(Vr)
        F = leform.assembly()

        return A, F
    
    @linear_system.register('penalty')
    def linear_system(self, mesh, p):
        """
        """
        self.space= LagrangeFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")
        
        D = ScalarDiffusionIntegrator(coef=1, q=self.p+2)
        M = ScalarMassIntegrator(coef=-self.k**2, q=self.p+2)
        R = ScalarRobinBCIntegrator(coef=self.k * 1j, q=self.p+2)
        G = JumpPenaltyIntergrator(coef=self.gamma, q=self.p+2)
 
        beform = BilinearForm(self.space)
        beform.add_integrator(D)
        beform.add_integrator(M)
        beform.add_integrator(R)
        beform.add_integrator(G)
        A = beform.assembly() 

        f = ScalarSourceIntegrator(self.pde.source, q=self.p+2)
        Vr = ScalarRobinSourceIntegrator(self.pde.robin, q=self.p+2)

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
        A, F = self.linear_system[f"{self.method}"](self.mesh, self.p)
        uh = self.space.function(dtype=bm.complex128)
        uh[:] = self.solve[self.solver](A, F)
        error = self.mesh.error(self.pde.solution, uh.value)

        node = self.mesh.node
        x, y = node[:, 0], node[:, 1]
        u_exact = self.pde.solution(node) 

        z1 = uh.real
        z2 = u_exact.real

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_trisurf(x, y, z1, cmap='viridis', linewidth=0.2)
        ax1.set_title(f"{self.method} Numerical Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("Re(u)")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_trisurf(x, y, z2, cmap='plasma', linewidth=0.2)
        ax2.set_title("Exact Solution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("Re(u)")

        plt.tight_layout()
        plt.show()

        return uh, error
        
    @run.register('error')
    def run(self):
        """
        Plot:
        1. Pointwise error (|uh - u_exact|) of Standard FEM on 3D surface
        2. Pointwise error of Interior Penalty FEM on 3D surface
        3. Bar chart comparing relative L2 errors
        """
        node = self.mesh.node
        x, y = node[:, 0], node[:, 1]
        u_exact = self.pde.solution(node)

        # -------- Standard FEM --------
        self.method = "standard"
        A, F = self.linear_system[self.method](self.mesh, self.p)
        uh = self.space.function(dtype=bm.complex128)
        uh[:] = self.solve[self.solver](A, F)
        err_std = self.mesh.error(self.pde.solution, uh.value)
        pointwise_err_std = bm.abs(uh - u_exact)

        # -------- Interior Penalty FEM --------
        self.method = "penalty"
        A, F = self.linear_system[self.method](self.mesh, self.p)
        val = self.space.function(dtype=bm.complex128)
        val[:] = self.solve[self.solver](A, F)
        err_ip = self.mesh.error(self.pde.solution, val.value)
        pointwise_err_ip = bm.abs(val - u_exact)

        # -------- Plotting --------
        fig = plt.figure(figsize=(18, 5))

        # ---- Plot 1: Pointwise Error of Standard FEM ----
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_trisurf(x, y, pointwise_err_std.real, cmap='viridis', linewidth=0.2)
        ax1.set_title("Pointwise Error (Standard FEM)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("|u - uh|")

        # ---- Plot 2: Pointwise Error of Interior Penalty FEM ----
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_trisurf(x, y, pointwise_err_ip.real, cmap='plasma', linewidth=0.2)
        ax2.set_title("Pointwise Error (Interior Penalty FEM)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("|u - val|")

        # ---- Plot 3: L2 Relative Error Comparison ----
        ax3 = fig.add_subplot(133)
        methods = ['Standard FEM', 'Interior Penalty FEM']
        errors = [err_std, err_ip]
        ax3.bar(methods, errors, color=['skyblue', 'salmon'])
        ax3.set_yscale('log')
        ax3.set_ylabel('Relative $L^2$ Error (log scale)')
        ax3.set_title(f'Error Comparison (k = {self.k})')

        plt.tight_layout()
        plt.show()

