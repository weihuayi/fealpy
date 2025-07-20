from typing import Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..model import PDEModelManager, ComputationalModel
from ..model.diffusion import DiffusionPDEDataT
from ..decorator import variantmethod

# FEM imports
from ..functionspace import LagrangeFESpace, TensorFunctionSpace
from ..fem import BilinearForm, LinearForm
from ..fem import ScalarDiffusionIntegrator, ScalarSourceIntegrator

class DiffusionLFEMModel(ComputationalModel):
    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEModelManager("diffusion")

    def set_pde(self, pde: Union[DiffusionPDEDataT, int] = 1) -> None:
        if isinstance(pde, int):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, meshtype: str = "uniform_tri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def set_diffusion_coef_type(self, coeftype: str) -> None:
        self.diffusion_coef_type = coeftype

    def linear_system(self, mesh, p):
        GD = mesh.geo_dimension()
        self.sspace= LagrangeFESpace(mesh=mesh, p=p, ctype='C')
        self.tspace = TensorFunctionSpace(scalar_space=self.sspace, shape=(GD, -1))

        TLDOF = self.tspace.number_of_local_dofs()
        TGDOF = self.tspace.number_of_global_dofs()
        self.logger.info(f"local DOFs: {TLDOF}, global DOFs: {TGDOF}")
        self.uh = self.tspace.function()

        bform = BilinearForm(self.tspace)

        if self.diffusion_coef_type == "piecewise_constant":
            ispace = LagrangeFESpace(mesh=mesh, p=0, ctype='D')
            diffusion_coef = ispace.interpolate(u=self.pde.diffusion_coef)
        elif self.diffusion_coef_type == "continuous":
            ispace1 = LagrangeFESpace(mesh=mesh, p=2, ctype='C')
            diffusion_coef = ispace1.interpolate(u=self.pde.diffusion_coef)

        DI = ScalarDiffusionIntegrator(coef=diffusion_coef, q=p+3)
        bform.add_integrator(DI)

        lform = LinearForm(self.tspace)
        SI = ScalarSourceIntegrator(self.pde.source)
        lform.add_integrator(SI)

        A = bform.assembly()
        F = lform.assembly()

        return A, F

    def apply_bc(self, A, F):
        from ..fem import DirichletBC

        if hasattr(self.pde, 'dirichlet'):
            A, F = DirichletBC(
                    self.tspace,
                    gd=self.pde.dirichlet,
                    threshold=self.pde.is_dirichlet_boundary).apply(A, F)
        else:
            A, F = DirichletBC(self.tspace, gd=self.pde.solution).apply(A, F)

        return A, F

    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import spsolve
        return spsolve(A, F, solver='mumps')


    @variantmethod('onestep')
    def run(self):
        A, F = self.linear_system(self.mesh, self.p)
        A, F = self.apply_bc(A, F)
        self.uh[:] = self.solve(A, F)
        l2, h1 = self.postprocess()

        self.logger.info(f"L2 Error: {l2},  H1 Error: {h1}.")

    @run.register('uniform_refine')
    def run(self, maxit=5):
        errorType = ['$|| \\boldsymbol{u}_h - \\boldsymbol{u} ||_{L^2}$', '$|| \\boldsymbol{u}_h - \\boldsymbol{u} ||_{H^1}$']
        errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
        NDof = bm.zeros(maxit, dtype=bm.int32)
        h = bm.zeros(maxit, dtype=bm.float64)
    
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2, h1 = self.postprocess()

            errorMatrix[0, i] = l2
            errorMatrix[1, i] = h1

            self.logger.info(f"{i}-th step with  L2 Error: {l2},  H1 Error: {h1}.")

            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()

            NDof[i] = self.tspace.number_of_global_dofs()

        print("errorMatrix:\n", errorType, "\n", errorMatrix)
        print("NDof:", NDof)
        order_l2 = bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:])
        order_h1 = bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:])
        print("Order of convergence in L2 norm:", order_l2)
        print("Order of convergence in H1 norm:", order_h1)
        print("---------------")

    @variantmethod("error")
    def postprocess(self):
        l2 = self.mesh.error(self.pde.solution, self.uh)
        h1 = self.mesh.error(self.pde.gradient, self.uh.grad_value)

        return l2, h1
