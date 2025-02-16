
from ..backend import backend_manager as bm

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC

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

        self.p = p
        self.pde = pde
        self.mesh = mesh
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


    def cg_solve(self):
        """
        """
        from ..solver import cg 
        self.uh[:] = cg(self.A, self.b, maxiter=5000, atol=1e-14, rtol=1e-14)
        if self.timer is not None:
            self.timer.send(f"求解 Poisson 方程线性系统")

        return self.uh

    def gs_solve(self):
        from ..solver import gs

        self.uh[:], info = gs(self.A, self.b, maxiter=200, rtol=1e-8)
        if self.timer is not None:
            self.timer.send(f"Gausss Seidel 方法求解 Poisson 方程线性系统")
        if self.logger is not None:
            self.logger.info(f"GS solver with {info['niter']} iterations"
                             f" and relative residual {info['residual']:.4e}")
    
    def jacobi_solve(self):
        from ..solver import jacobi

        self.uh[:], info = jacobi(self.A, self.b, maxiter=200, rtol=1e-8)
        if self.timer is not None:
            self.timer.send(f"Jacobi 方法求解 Poisson 方程线性系统")
        if self.logger is not None:
            self.logger.info(f"Jacobi solver with {info['niter']} iterations"
                             f" and relative residual {info['residual']:.4e}")


    def gamg_solve(self, P=None, cdegree=[1]):
        """
        """
        from ..solver import GAMGSolver
        solver = GAMGSolver(isolver='MG') 

        solver.setup(self.A, P=P, space=self.space, cdegree=cdegree)

        x,info = solver.solve(self.b)
        self.uh[:] = x 
            
        res = info['residual']
        res_0 = bm.linalg.norm(self.b)
        stop_res = res/res_0

        # if self.timer is not None:
        #     self.timer.send(f"GAMG 方法求解 Poisson 方程线性系统")
        # if self.logger is not None:
        #     self.logger.info(f"GAMG solver with {info['niter']} iterations"
        #                      f" and relative {info['residual']:.4e}")
        # # 检查收敛状态
        # if stop_res <= rtol:
        #     if self.logger is not None:
        #         self.logger.info(
        #             f"GAMG solver converged: stop_res = {stop_res:.2e} <= rtol = {rtol:.2e}"
        #         )
        #     converged = True
        # else:
        #     if self.logger is not None:
        #         self.logger.warning(
        #             f"GAMG solver NOT converged: stop_res = {stop_res:.2e} > rtol = {rtol:.2e}"
        #         )
        #     converged = False

        # 返回解和收敛标志
        return x,info


    def show_mesh_and_solution(self):
        """
        """
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        fig = plt.figure()
        axes = fig.add_subplot(121)
        mesh.add_plot(axes)
        axes = fig.add_subplot(122, projection='3d')
        axes.plot_trisurf(node[:, 0], node[:, 1], self.uh, triangles=cell, cmap='rainbow')
        plt.show()


    def L2_error(self):
        """
        """
        return self.mesh.error(self.pde.solution, self.uh)



        


