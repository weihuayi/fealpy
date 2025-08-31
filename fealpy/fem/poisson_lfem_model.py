from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEModelManager, ComputationalModel
from ..model.poisson import PoissonPDEDataT
from ..decorator import variantmethod

# FEM imports
from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarDiffusionIntegrator, ScalarSourceIntegrator
from  . import RecoveryAlg

class PoissonLFEMModel(ComputationalModel):
    """
    """
    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEModelManager("poisson")

    def set_pde(self, pde: Union[PoissonPDEDataT, str]="coscos"):
        """
        """
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, meshtype: str = "tri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")


    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def linear_system(self, mesh, p):
        """
        """

        self.space= LagrangeFESpace(mesh, p=p)

        LDOF = self.space.number_of_local_dofs()
        GDOF = self.space.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")
        self.uh = self.space.function()

        bform = BilinearForm(self.space)
        DI = ScalarDiffusionIntegrator()
        bform.add_integrator(DI)

        lform = LinearForm(self.space)
        SI = ScalarSourceIntegrator(self.pde.source)
        lform.add_integrator(SI)

        A = bform.assembly()
        F = lform.assembly()
        return A, F

    def apply_bc(self, A, F):
        """
        Apply boundary conditions to the linear system.
        """
        from ..fem import DirichletBC
        if hasattr(self.pde, 'dirichlet'):
            A, F = DirichletBC(
                    self.space,
                    gd=self.pde.dirichlet,
                    threshold=self.pde.is_dirichlet_boundary).apply(A, F)
        else:
            A, F = DirichletBC(self.space, gd=self.pde.solution).apply(A, F)
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

    @solve.register('pcg')
    def solve(self, A, F):
        pass


    @variantmethod('onestep')
    def run(self):
        """
        """
        A, F = self.linear_system(self.mesh, self.p)
        A, F = self.apply_bc(A, F)
        self.uh[:] = self.solve(A, F)
        l2, h1 = self.postprocess()
        self.logger.info(f"L2 Error: {l2},  H1 Error: {h1}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()

    @run.register("h1_afem")
    def run(self, theta=0.95, maxit=6):
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                error = self.mesh.error(self.pde.gradient, 
                                        self.uh.grad_value,
                                        celltype=True) ** 2
                indice = bm.argsort(-error)  
                cum_ratio = bm.cumsum(error[indice], axis=0) / bm.sum(error)
                mask = cum_ratio >= theta**2
                idx = bm.nonzero(mask)[0][0] + 1
                isMark = bm.zeros_like(indice, dtype=bm.bool)
                isMark = bm.set_at(isMark, indice[:idx], True)
                self.mesh.bisect(isMark)

    @run.register("res_afem")
    def run(self, theta=0.95, maxit=6):
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                bcs = bm.array([[1/3,1/3,1/3]], dtype=bm.float64)
                nu = self.mesh.edge_unit_normal()                
                grad = self.uh.grad_value(bcs)
                edge2cell = self.mesh.edge_to_cell()
                res_grad = grad[edge2cell[:, 0]] - grad[edge2cell[:, 1]]
                dot = bm.einsum('cqn, cn -> cq', res_grad, nu).reshape(-1)
                len = self.mesh.entity_measure('edge')
                bd_edge = self.mesh.boundary_edge_flag()
                edge_error = (dot * len)**2 * ~bd_edge         
                cell2edge = self.mesh.cell_to_edge()
                error = bm.sum(edge_error[cell2edge], axis=-1) 
                indice = bm.argsort(-error)  
                cum_ratio = bm.cumsum(error[indice], axis=0) / bm.sum(error)
                mask = cum_ratio >= theta**2
                idx = bm.nonzero(mask)[0][0] + 1
                isMark = bm.zeros_like(indice, dtype=bm.bool)
                isMark = bm.set_at(isMark, indice[:idx], True)
                self.mesh.bisect(isMark)

    @run.register("recovery_afem")
    def run(self, theta=0.95, maxit=6, method='simple'):
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                alg = RecoveryAlg()
                error = alg.recovery_estimate(self.uh, method=method)
                indice = bm.argsort(-error)  
                cum_ratio = bm.cumsum(error[indice], axis=0) / bm.sum(error)
                mask = cum_ratio >= theta**2
                idx = bm.nonzero(mask)[0][0]
                isMark = bm.zeros_like(indice, dtype=bm.bool)
                isMark = bm.set_at(isMark, indice[:idx], True)
                self.mesh.bisect(isMark)

    @variantmethod("error")
    def postprocess(self):
        """
        """
        l2 = self.mesh.error(self.pde.solution, self.uh)
        h1 = self.mesh.error(self.pde.gradient, self.uh.grad_value)
        return l2, h1
