from typing import Optional, Union
from ..backend import bm
from ..model import PDEModelManager
from ..model.elliptic import EllipticPDEDataT
from ..decorator import variantmethod

class PolyharmonicCrFEMModel:
    """
        Smooth Finite Element Method to solve the equation \(\Delta^{m+1} u =
        f\).  on a 2D/3D domain, using C^m-conforming finite element spaces.
    """
    def __init__(self, logger=None, timer=None):
        if logger is None:
            from .. import logger 

        self.logger = logger
        self.logger.setLevel('WARNING')

        if timer is None:
            from ..utils import timer
        self.timer = timer
        self.pdm = PDEModelManager("elliptic")

    def set_pde(self, pde: Union[EllipticPDEDataT, str]="biharm2d"):
        """
        Assign the PDE problem to be solved.

        Args:
            pde: Either a string key for predefined examples (e.g., "biharm2d",
            "triharm2d, "biharm3d"")
                 or a user-defined EllipticPDEDataT object.
        """
        if isinstance(pde, str):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde
        
    def set_init_mesh(self, **kwargs):
        self.mesh = self.pde.init_mesh(**kwargs)

    def set_order(self, p: int = 5):
        """
        Set the polynomial degree of the finite element space.

        Args:
            p: Polynomial degree. Should satisfy p >= 2^dm + 1 for stability and approximation.
        """
        self.p = p

    def set_smoothness(self, m: int = 1):
        """
        Set the smoothness order of the FEM space.

        Args:
            m: Desired smoothness order. Corresponds to solving Δ^{m+1} u = f.
        """
        self.m = m



    def linear_system(self, mesh, p):
        """
        Assemble the finite element linear system (stiffness matrix and load vector).

        Returns:
            A: Sparse matrix representing the bilinear form.
            F: Load vector.
        """
        from ..functionspace import CmConformingFESpace2d
        from ..functionspace import CmConformingFESpace3d
        from ..fem import BilinearForm, ScalarDiffusionIntegrator
        from ..fem import MthLaplaceIntegrator
        from ..fem import LinearForm, ScalarSourceIntegrator
        m = self.pde.order()
        
        if self.mesh.TD == 2:
            self.space = CmConformingFESpace2d(mesh, p, m)
        if self.mesh.TD == 3:
            self.space = CmConformingFESpace3d(mesh, p, m)
        self.uh = self.space.function() # 建立一个有限元函数

        bform = BilinearForm(self.space)
        integrator = MthLaplaceIntegrator(m=m+1, coef=1, q=p+4)
        bform.add_integrator(integrator)

        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=p+4))

        A = bform.assembly()
        F = lform.assembly()

        return A, F

    def apply_bc(self, A, F):
        """
        Apply Dirichlet boundary conditions.

        Returns:
            A, F: Modified matrix and vector after applying boundary conditions.
        """
        from ..fem import DirichletBC
        A, F = DirichletBC( self.space, gd=self.pde.get_flist()).apply(A, F)
        return A, F

    @variantmethod("direct")
    def solve(self, A, F):
        """
        Solve the linear system using a direct solver (default method).
        """
        from ..solver import spsolve
        return spsolve(A, F, solver='scipy')


    @solve.register('amg')
    def solve(self, A, F):
        pass

    @solve.register('mumps')
    def solve(self, A, F):
        from ..solver import spsolve
        return spsolve(A, F, solver='mumps')

    @solve.register('pcg')
    def solve(self, A, F):
        pass


    @variantmethod('onestep')
    def run(self):
        """
        Execute one step of the solve process:
        - Assemble system
        - Apply boundary conditions
        - Solve the linear system
        - Postprocess and report errors
        """
        A, F = self.linear_system(self.mesh, self.p)
        #self.timer.send(f"组装方程离散系统") 
        A, F = self.apply_bc(A, F)
        #self.timer.send(f"处理边界条件")
        self.uh[:] = self.solve(A, F)
        #self.timer.send(f"求解线性系统")
        l2, h1, h2 = self.postprocess()
        print(f"L2 error: {l2}, H1 error: {h1}")
        #self.logger.info(f"L2 error: {l2}, H1 error: {h1}")

    @run.register('uniform_refine')
    def run(self, maxit=4):
        pass


    @run.register("bisect")
    def run(self):
        pass


    @variantmethod("error")
    def postprocess(self):
        """
        Compute L2, H1, and H2 errors of the numerical solution.

        Returns:
            l2: L2-norm error between exact and numerical solution.
            h1: H1-seminorm error.
            h2: H2-seminorm error (approximate if pde.hessian is provided).
        """
        from ..decorator import barycentric

        @barycentric
        def ugval(p):
            return self.space.grad_m_value(self.uh, p, 1)

        @barycentric
        def ug2val(p):
            return self.space.grad_m_value(self.uh, p, 2)

        l2 = self.mesh.error(self.pde.solution, self.uh)
        h1 = self.mesh.error(self.pde.gradient, ugval)
        h2 = self.mesh.error(self.pde.hessian, ug2val)
        return l2, h1, h2


