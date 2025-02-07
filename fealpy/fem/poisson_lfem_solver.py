
from ..backend import backend_manager as bm

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC
from ..solver import cg


class PoissonLFEMSolver:
    """
    """
    def __init__(self, pde, mesh, p, timer=None, logger=None):
        """
        """
        # 计时与日志
        self.timer = timer
        self.logger = logger

        self.pde = pde
        self.space= LagrangeFESpace(mesh, p=p)
        self.uh = self.space.function() # 建立一个有限元函数
        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source))
        A = bform.assembly()
        b = lform.assembly()
        if self.timer is not None:
            self.timer.send(f"组装 Poisson 方程离散系统")

        gdof = self.space.number_of_global_dofs()
        self.A, self.b = DirichletBC(self.space, gd=pde.solution).apply(A, b)
        if self.timer is not None:
            self.timer.send(f"处理 Poisson 方程 D 边界条件")


    def solve(self):
        """
        """
        self.uh[:] = cg(self.A, self.b, maxiter=5000, atol=1e-14, rtol=1e-14)
        if self.timer is not None:
            self.timer.send(f"求解 Poisson 方程线性系统")

        return self.uh


    def gamg_solve(self, P):
        """
        """
        from ..solver import GAMGSolver
        solver = GAMGSolver() 

        solver.A = [self.A]
        solver.P = P
        solver.R = []
        for m in P:
            s = m.sum(axis=1)
            # m.T/s[None, :]
            solver.R.append(m.T.div(s))
            print("R.shape", solver.R[-1].shape)
            print("A.shape", solver.A[-1].shape)
            a = solver.R[-1].matmul(solver.A[-1])
            solver.A.append(a.matmul(m))



