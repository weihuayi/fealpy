from typing import Optional, Union
from scipy.sparse import coo_array, bmat

from ..backend import backend_manager as bm
from ..model import PDEModelManager, ComputationalModel
from ..model.surface_poisson import SurfacePDEDataT
from ..decorator import variantmethod

from ..mesh import Mesh
from ..functionspace.parametric_lagrange_fe_space import ParametricLagrangeFESpace
from ..fem import (BilinearForm, 
                   ScalarDiffusionIntegrator,
                   LinearForm, 
                   ScalarSourceIntegrator)
from ..sparse import COOTensor

from ..solver import spsolve, cg


class SurfacePoissonLFEMModel(ComputationalModel):
    """A class to represent a surface Poisson problem using the Lagrange finite element method (LFEM).
    
    Attributes:
        mesh: The mesh of the domain.   
    Methods:
        set_pde(pde): Initializes the PDE model for the problem.
        set_mesh(mesh): Sets the mesh for the domain.
        set_space_degree(p): Sets the polynomial degree for the isoparametric finite element space.
        surface_poisson_system(): Assembles the linear system for the Poisson problem.
        solve(): Solves the system using the selected solver method.
        run(): Runs the solver, iterating over the solution process.
        postprocess(): Computes the L2 error of the solution.                                        
    Reference:
        https://wnesm678i4.feishu.cn/wiki/SsOKwQiVqi241WkusA9cn9ylnPf
    """
    
    def __init__(self, options):
       self.options = options
       super().__init__(pbar_log=options['pbar_log'], 
                        log_level=options['log_level'])
       self.set_pde(options['pde'])
       self.pde.init_mesh.set(options['init_mesh'])
       mesh = self.pde.init_mesh(options['mesh_degree'])
       self.set_mesh(mesh)
       self.set_space_degree(options['space_degree']) 
    
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing the configuration of the current surface Poisson problem finite element model.
        
        Returns
            str: A multi-line string containing the current model configuration, 
            displaying information such as the mesh, PDE, space degree, solver.
        """
        s = f"{self.__class__.__name__}(\n"
        s += f"  pde            : {self.pde.__class__.__name__}\n"
        s += f"  mesh           : {self.mesh.__class__.__name__}\n"
        s += f"  space_degree   : {self.p}\n"
        s += ")"

        self.logger.info(f"\n{s}")
        return s
  
    def set_pde(self, pde: Union[SurfacePDEDataT, int] = 1) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager("surface_poisson").get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def set_space_degree(self, p: int) -> None:
        self.p = p

    def surface_poisson_system(self):
        """Construct the linear system for the surface problem.

        Returns
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
        """Solve the surface problem using the finite element method."""
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
