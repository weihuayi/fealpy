from typing import Optional, Union
from scipy.sparse import coo_array, csr_matrix, bmat
from ..backend import bm
from ..model import PDEDataManager, ComputationalModel
from ..model.surface_poisson import SurfacePDEDataT
from ..decorator import variantmethod

from ..mesh import Mesh
from ..functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..sparse import COOTensor

from ..solver import spsolve, cg


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
       self.set_init_mesh(options['mesh_degree'], options['init_mesh']) 
       self.set_space_degree(options['space_degree']) 
    
    def set_pde(self, pde: Union[SurfacePDEDataT, str] = "sphere"):
        if isinstance(pde, str):
            self.pde = PDEDataManager("surface_poisson").get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, p:int, mesh:Union[Mesh, str] = "ltri", **kwargs):
        if isinstance(mesh,str):
            self.mesh = self.pde.init_mesh[mesh](p, **kwargs)
        else:
            self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        #fname = f"sphere_test.vtu"
        #self.mesh.to_vtk(fname=fname)
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int) -> None:
        self.p = p

    def surface_poisson_system(self):
        """
        Construct the linear system for the surface problem.

        Returns:
            The diffusion matrix and source matrix.
        """
        self.space = ParametricLagrangeFESpace(self.mesh, self.p)
        self.uh = self.space.function()

        bform = BilinearForm(self.space)
        bform.add_integrator(ScalarDiffusionIntegrator(method='isopara'))
        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, method='isopara'))
        
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
    def solve(self):
        """
        Solve the surface problem using the finite element method.
        """
        A, F = self.surface_poisson_system()
        return spsolve(A, F, solver='scipy')
    
    @solve.register('cg')
    def solve(self):
        A, F = self.surface_poisson_system()
        x = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14).reshape(-1)
        self.uh[:] = x[:-1]  # Exclude the last element which
        l2 = self.postprocess()
        self.logger.info(f"L2 Error: {l2}.")
    
    @variantmethod('onestep')
    def run(self):
        self.uh[:] = self.solve()[:-1]
        l2 = self.postprocess()
        self.logger.info(f"L2 Error: {l2}.")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        for i in range(maxit):
            A,F = self.surface_poisson_system()
            self.uh[:] = self.solve()[:-1]
            l2 = self.postprocess()
            self.logger.info(f"{1}-th step with  L2 Error: {l2}.")
            if i < maxit - 1:
                self.logger.info(f"Refining mesh {i+1}/{maxit}.")
                self.mesh.uniform_refine()

    @variantmethod("error")
    def postprocess(self):
        l2 = self.mesh.error(self.pde.solution, self.uh)
        return l2
