from ..backend import backend_manager as bm
from fealpy.decorator import barycentric
from ..functionspace import CmConformingFESpace2d
from ..functionspace import LagrangeFESpace
from . import BilinearForm, ScalarDiffusionIntegrator
from .mthlaplace_integrator import MthLaplaceIntegrator
from . import LinearForm, ScalarSourceIntegrator
from . import DirichletBC
from ..solver import spsolve
from fealpy.pde.biharmonic_triharmonic_2d import DoubleLaplacePDE, TripleLaplacePDE, get_flist


class MthLaplaceSmoothFEMSolver:
    """
     光滑元方法计算 \(\Delta^{m+1} u = f\) 方程
    """
    def __init__(self, pde, mesh, p, m, timer=None, logger=None):
        """
        @param pde:
            The partial differential equation to solve.

        @param mesh:
            The finite element mesh, specifically for a 2D triangular mesh.

        @param m:
            The type of PDE to solve:
            - m = 1: Double Laplace equation (双调和方程)
            - m = 2: Triple Laplace equation (三调和方程)

        @param p:
            The degree of the finite element space. Should satisfy:
            - p >= 4 * m + 1

        Example:
            For the Double Laplace equation:
                pde = DoubleLaplacePDE(u)
            For the Triple Laplace equation:
                pde = TripleLaplacePDE(u)

        Where:
            u = sp.sin(2 * sp.pi * y) * sp.sin(2 * sp.pi * x)
            x = sp.symbols('x')
            y = sp.symbols('y')
        """
        # 计时与日志
        self.timer = timer
        self.logger = logger
        self.mesh = mesh

        self.pde = pde
        ulist = get_flist(pde.su)[:2*m+1]

        self.space = CmConformingFESpace2d(mesh, p, m)
        self.uh = self.space.function() # 建立一个有限元函数
        bform = BilinearForm(self.space)
        integrator = MthLaplaceIntegrator(m=m+1, coef=1, q=p+4)
        bform.add_integrator(integrator)
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=p+4))
        A = bform.assembly()
        b = lform.assembly()
        if self.timer is not None:
            self.timer.send(f"组装 Poisson 方程离散系统")

        gdof = self.space.number_of_global_dofs()
        self.A, self.b = DirichletBC(self.space, gd=ulist).apply(A, b)
        if self.timer is not None:
            self.timer.send(f"处理 Poisson 方程 D 边界条件")


    def solve(self):
        """
        """
        self.uh[:] = spsolve(self.A, self.b)
        if self.timer is not None:
            self.timer.send(f"求解 Poisson 方程线性系统")

        return self.uh
    def error(self):
        uh = self.uh
        mesh = self.mesh
        pde = self.pde
        space = self.space

        @barycentric
        def ugval(p):
            return space.grad_m_value(uh, p, 1)

        @barycentric
        def ug2val(p):
            return space.grad_m_value(uh, p, 2)
        error = bm.zeros(3, dtype=uh.dtype)
        error[0] = mesh.error(pde.solution, uh)
        error[1] = mesh.error(pde.gradient, ugval)
        error[2] = mesh.error(pde.hessian, ug2val)
        return error



