
from .. import logger
logger.setLevel('WARNING')
from typing import Union
from ..backend import bm
from ..model import PDEDataManager, ComputationalModel
from ..model.helmholtz import HelmholtzPDEDataT

from ..mesh import Mesh
from ..functionspace import LagrangeFESpace
from ..fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarRobinBCIntegrator 
from ..fem import HelmipMassIntegrator  
from ..fem import ScalarSourceIntegrator, ScalarRobinSourceIntegrator         
from ..fem import BilinearForm, LinearForm
from ..solver import spsolve

from ..utils import timer
import matplotlib.pyplot as plt


class  HelmInteriorPenaltyLFEMModel(ComputationalModel):
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
                - method (str): FEM method, 'standard' or 'interior_penalty'.
        """
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['nx'], ny=options['ny'])
        self.set_space_degree(options['space_degree'])
        self.set_wave_number(options['wave_number'])
        self.set_gamma(options['gamma'])
        self.set_method(options['method'])

    def set_pde(self, pde:Union[HelmholtzPDEDataT, str]  = "bessel2d"):
        """
        Set the PDE model for the problem.

        Args:
            pde: PDE model instance or string key to get example PDE.
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('helmholtz').get_example(pde)
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

    def set_method(self, method: str = 'standard'):
        """
        Set the finite element method type.

        Args:
            method: 'standard' for classical FEM or 'interior_penalty' for IP-FEM.

        """
        if method not in ['standard', 'interior_penalty']:
            raise ValueError("Invalid method, must be 'standard' or 'interior_penalty'")
        self.method = method

    def solution(self):
        """
        Solve the Helmholtz problem using the specified FEM method.

        Returns:
            uh: Numerical solution as a finite element function.
            error: Relative L2 error between numerical and exact solutions.
        """
        self.pde.set(k=self.k)
        space = LagrangeFESpace(self.mesh, self.p)
        
        tmr = timer()
        next(tmr)
        D = ScalarDiffusionIntegrator(coef=1, q=self.p+2)
        M = ScalarMassIntegrator(coef=-self.k**2, q=self.p+2)
        R = ScalarRobinBCIntegrator(coef=self.k * 1j, q=self.p+2)
 
        beform = BilinearForm(space)
        beform.add_integrator(D)
        beform.add_integrator(M)
        beform.add_integrator(R)

        if self.method == 'interior_penalty':
            G = HelmipMassIntegrator(coef=self.gamma, q=self.p+2)
            beform.add_integrator(G)
        tmr.send(f'矩组装时间为')
        A = beform.assembly() 

        f = ScalarSourceIntegrator(self.pde.source, q=self.p+2)
        Vr = ScalarRobinSourceIntegrator(self.pde.robin, q=self.p+2)

        leform = LinearForm(space)
        leform.add_integrator(f)
        leform.add_integrator(Vr)
        F = leform.assembly()
        tmr.send(f'向量组装时间')

        uh = space.function(dtype=bm.complex128)
        uh[:] = spsolve(A, F,"scipy")
        tmr.send(f'求解器时间')
        next(tmr)

        error = self.mesh.error(self.pde.solution, uh.value)

        return uh, error
    
    def plot_solution(self):
        """
        Plot the numerical solution and exact solution in two 3D subplots.
        """        
        self.pde.set(k=self.k)
        node = self.mesh.node
        x, y = node[:, 0], node[:, 1]
        u_exact = self.pde.solution(node) 

        z1,_ = self.solution()
        z1 = z1.real
        z2 = u_exact.real

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_trisurf(x, y, z1, cmap='viridis', linewidth=0.2)
        ax1.set_title("Numerical Solution")
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

    
    def plot_error_comparison(self):
        """
        Compare errors of standard FEM and interior penalty FEM at fixed wave number.

        Draw a bar chart with relative L2 errors on a log scale.
        """
        k = self.k

        # -------- Standard FEM --------
        self.set_method('standard')
        _,err_std = self.solution()

        # -------- Interior Penalty FEM --------
        self.set_method('interior_penalty')
        _,err_ip = self.solution()

        methods = ['Standard FEM', 'Interior Penalty FEM']
        errors = [err_std, err_ip]

        plt.figure(figsize=(6, 5))
        plt.bar(methods, errors, color=['skyblue', 'salmon'])
        plt.yscale('log')
        plt.ylabel('Relative $L^2$ Error (log scale)')
        plt.title(f'Error Comparison (k = {k})')
        plt.tight_layout()
        plt.show()
