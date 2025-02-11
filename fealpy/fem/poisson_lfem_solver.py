
from ..backend import backend_manager as bm

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC
from fealpy.solver import cg,gamg_solver,GaussiSeidei

from fealpy.sparse import csr_matrix



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
        gs,res_gs,iter_gs = GaussiSeidei.gs(self.A,self.b)
        if self.timer is not None:
            self.timer.send(f"求解 Poisson 方程线性系统")

        return self.uh,gs


    def gamg_solve(self, P, ptype: str='V',level=0,rtol: float=1e-8):
        """
        """
        from ..solver import GAMGSolver
        solver = GAMGSolver() 

        solver.A = [self.A]
        solver.P = P
        solver.R = []
        solver.L = [self.A.tril()]
        solver.U = [self.A.triu()]
        solver.ptype = ptype
        for m in P:
            # s = m.sum(axis=1)
            # m.T/s[None, :]
            # solver.R.append(m.T.div(s))
            solver.R.append(m.T)
            a = solver.R[-1].matmul(solver.A[-1])
            solver.A.append(a.matmul(m))
            solver.L.append(solver.A[-1].tril())
            solver.U.append(solver.A[-1].triu())

        # if solver.ptype == 'V':
        #     x =  solver.vcycle(self.b)
        # elif solver.ptype == 'W':
        #     x = solver.wcycle(self.b)
        # elif solver.ptype == 'F':
        #     x = solver.fcycle(self.b)
        x = solver.solve(self.b)
            
        res = solver.A[0].matmul(x) - self.b
        res = bm.sqrt(bm.sum(res**2))
        res_0 = bm.sqrt(bm.sum(self.b**2))

        stop_res = res/res_0
        # 输出 stop_res
        if self.logger is not None:
            self.logger.info(f"stop_res = {stop_res:.2e}")
        else:
            print(f"stop_res = {stop_res:.2e}")

        # 检查收敛状态
        if stop_res <= rtol:
            if self.logger is not None:
                self.logger.info(
                    f"GAMG solver converged: stop_res = {stop_res:.2e} <= rtol = {rtol:.2e}"
                )
            converged = True
        else:
            if self.logger is not None:
                self.logger.warning(
                    f"GAMG solver NOT converged: stop_res = {stop_res:.2e} > rtol = {rtol:.2e}"
                )
            converged = False

        # 返回解和收敛标志
        return x, converged



        


