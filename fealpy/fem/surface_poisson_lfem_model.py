from typing import Optional, Union
from scipy.sparse import coo_array, bmat
from ..backend import backend_manager as bm
from ..model import PDEDataManager, ComputationalModel
from ..model.surface_poisson import SurfacePDEDataT
from ..decorator import variantmethod


from ..functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..sparse import COOTensor


class SurfacePoissonLFEMModel(ComputationalModel):
    """
    A class to represent a surface Poisson problem using the Lagrange finite element method (LFEM).
    
    Attributes:
        mesh: The mesh of the domain.                                           
    Reference:
        https://wnesm678i4.feishu.cn/wiki/SsOKwQiVqi241WkusA9cn9ylnPf
    """
    
    
    def __init__(self, options):
       self.options = options
       super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
       self.set_pde(options['pde'])
       self.set_init_mesh(options['init_mesh']) 
       self.set_space_degree(options['space_degree']) 
       

    def set_pde(self, pde: Union[SurfacePDEDataT, str] = "sphere"):
        if isinstance(pde, str):
            self.pde = PDEDataManager("surface_poisson").get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, meshtype: str = "ltri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def surface_poisson_system(self, mesh, p):
        self.space = ParametricLagrangeFESpace(mesh, p=p)
        self.uh = self.space.function()

        bform = BilinearForm(self.space)
        SDI = ScalarDiffusionIntegrator(method='isopara')
        bform.add_integrator(SDI)

        lform = LinearForm(self.space)
        SSI = ScalarSourceIntegrator(self.pde.source, method='isopara')
        lform.add_integrator(SSI)

        A = bform.assembly(format='coo')
        F = lform.assembly()

        C = self.space.integral_basis()

        def coo(A):
            data = A._values
            indices = A._indices   
            return coo_array((data, indices), shape=A.shape)
    
        A = bmat([[coo(A), C.reshape(-1,1)], [C[None,:], None]], format='coo')
        A = COOTensor(bm.stack([A.row, A.col], axis=0), A.data, spshape=A.shape)
        
        F = bm.concatenate((F, bm.array([0])))

        return A, F
    
    @variantmethod("direct")
    def solve(self, A, F):
        from ..solver import spsolve
        return spsolve(A, F, solver='scipy')
    
    @solve.register('cg')
    def solve(self, A, F):
        from ..solver import cg
        self.uh, info = cg(A, F, maxiter=5000, atol=1e-14, rtol=1e-14).reshape(-1)
        self.uh[:] = -self.uh[:-1]
        res = info['residual']
        self.logger.info(f"CG solver finished with residual: {res}")

    @variantmethod('onestep')
    def run(self):
        """
        """
        A, F = self.surface_poisson_system(self.mesh, self.p)
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

    @variantmethod("error")
    def postprocess(self):
        l2 = self.mesh.error(self.pde.solution, self.uh)
        return l2