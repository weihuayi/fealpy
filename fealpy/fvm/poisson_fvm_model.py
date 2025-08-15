from typing import Union
from ..backend import backend_manager as bm
from ..model import PDEModelManager, ComputationalModel
from ..functionspace import ScaledMonomialSpace2d
from ..fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ScalarCrossDiffusionIntegrator,
    DirichletBC,
)
from ..fem import (
    BilinearForm,
    LinearForm
)
from ..solver import spsolve

class PoissonFVMModel(ComputationalModel):
    """
    The Poisson equation in two-dimensional cases is solved by the finite volume method. 
    Through the iterative method, it has good applicability to various grid divisions.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "WARNING"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options["space_degree"])

    def set_pde(self, pde: Union[str, object]):
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, nx: int = 10, ny: int = 10):
        self.mesh = self.pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)    

    def set_space(self, degree: int = 0):
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
    
    def assemble_base_system(self):
        """组装基础系统矩阵 A 和源项 f"""
        A = BilinearForm(self.space)
        A.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        A = A.assembly()

        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet)
        A, f = dbc.DiffusionApply(A, f)
        return A, f

    def compute_cross_diffusion(self, uh):
        """计算交叉扩散项"""
        lform2 = LinearForm(self.space)
        lform2.add_integrator(ScalarCrossDiffusionIntegrator(uh, q=self.p + 2))
        return lform2.assembly()

    def solve(self, max_iter=6, tol=1e-6):
        """迭代对交叉扩散项的计算，并修正线性系统"""
        A, f = self.assemble_base_system()
        uh = spsolve(A, f)

        for i in range(max_iter):
            cross = self.compute_cross_diffusion(uh)
            rhs = f + cross
            uh_new = spsolve(A, rhs)
            err = bm.max(bm.abs(uh_new - uh))
            self.logger.info(f"[Iter {i+1}] residual = {err:.4e}")
            if err < tol:
                self.logger.info("Converged.")
                break
            uh = uh_new

        self.uh = uh
        return uh

    def compute_error(self):
        """计算 L2 误差"""
        cell_center = self.mesh.entity_barycenter('cell')
        u_exact = self.pde.solution(cell_center)
        error = bm.sqrt(bm.sum(self.mesh.entity_measure('cell') * (u_exact - self.uh)**2))
        return error

    def plot(self):
        import matplotlib.pyplot as plt
        cell_center = self.mesh.entity_barycenter('cell')  
        x, y = cell_center[:, 0], cell_center[:, 1]
        z_num = self.uh                     
        z_exact = self.pde.solution(cell_center)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_trisurf(x, y, z_num, cmap='viridis', linewidth=0.2)
        ax1.set_title("Numerical Solution (FVM)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u_h")
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_trisurf(x, y, z_exact, cmap='plasma', linewidth=0.2)
        ax2.set_title("Exact Solution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u_exact")

        plt.tight_layout()
        plt.show()


