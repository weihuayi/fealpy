from typing import Union
from ..backend import bm
from ..model import PDEDataManager, ComputationalModel
from ..model.linear_elasticity import LinearElasticityPDEDataT
from ..decorator import variantmethod

# FEM imports
from ..functionspace import LagrangeFESpace, TensorFunctionSpace
from ..fem import BilinearForm, LinearForm
from ..fem import LinearElasticityIntegrator, VectorSourceIntegrator
from ..material import LinearElasticMaterial

class LinearElasticityLFEMModel(ComputationalModel):
    def __init__(self):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEDataManager("linear_elasticity")

    def set_pde(self, pde: Union[LinearElasticityPDEDataT, str]="boxpoly"):
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, meshtype: str = "hex", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int = 1) -> None:    
        self.p = p

    def linear_system(self, mesh, p):
        self.space= LagrangeFESpace(mesh, p=p)
        self.tspace = TensorFunctionSpace(self.space, shape=(-1, 3))

        LDOF = self.tspace.number_of_local_dofs()
        GDOF = self.tspace.number_of_global_dofs()
        self.logger.info(f"local DOFs: {LDOF}, global DOFs: {GDOF}")
        self.uh = self.tspace.function()

        LEM = LinearElasticMaterial(
                                name='E1nu025',
                                lame_lambda=self.pde.lam(), shear_modulus=self.pde.mu(), 
                                hypo='3D', device=bm.get_device(self.uh[:])
                            )
        
        bform = BilinearForm(self.tspace)
        LEI = LinearElasticityIntegrator(
                                material=LEM, q=self.p+3, method=None
                            )
        bform.add_integrator(LEI)

        lform = LinearForm(self.tspace)
        SI = VectorSourceIntegrator(self.pde.body_force)
        lform.add_integrator(SI)

        A = bform.assembly()
        F = lform.assembly()

        return A, F

    def apply_bc(self, A, F):
        from ..fem import DirichletBC
        if hasattr(self.pde, 'displacement_bc'):
            A, F = DirichletBC(
                    self.tspace,
                    gd=self.pde.displacement_bc,
                    threshold=self.pde.is_displacement_boundary).apply(A, F)
        else:
            pass
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
        self.uh[:], info = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14, returninfo=True)
        res = info['residual']
        res_0 = bm.linalg.norm(F)
        stop_res = res/res_0
        self.logger.info(f"CG solver with {info['niter']} iterations"
                         f" and relative residual {stop_res:.4e}")

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
        l2 = self.postprocess()
        self.logger.info(f"L2 Error: {l2}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        for i in range(maxit):
            A, F = self.linear_system(self.mesh, self.p)
            A, F = self.apply_bc(A, F)
            self.uh[:] = self.solve(A, F)
            l2 = self.postprocess()
            self.logger.info(f"{i}-th step with  L2 Error: {l2}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()


    @run.register("bisect")
    def run(self):
        pass

    @variantmethod("error")
    def postprocess(self):
        l2 = self.mesh.error(self.pde.displacement, self.uh)
        return l2
