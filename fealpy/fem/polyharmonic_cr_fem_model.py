from typing import Optional, Union
from ..backend import bm
from ..model import PDEDataManager, ComputationalModel
from ..model.polyharmonic import PolyharmonicPDEDataT
from ..decorator import variantmethod

from ..mesh import Mesh
from ..functionspace import functionspace

from ..fem import BilinearForm, ScalarDiffusionIntegrator
from ..fem import PolyharmonicIntegrator
from ..fem import LinearForm, ScalarSourceIntegrator
from ..fem import DirichletBC


class PolyharmonicCrFEMModel(ComputationalModel):
    """
    Smooth Finite Element Method to solve the equation \(\Delta^{m+1} u = f\)
    on a 2D/3D domain, using \(C^m\)-conforming finite element spaces.

    Attributes:
        mesh: The computational mesh.
        pde: The PDE problem data.
        p: Polynomial degree of finite element space.
        m: Smoothness order.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['mesh_size'],
                           ny=options['mesh_size'] )
        self.set_space_degree(options['space_degree'])
        self.set_smoothness(options['smoothness'])

    def set_pde(self, pde: Union[PolyharmonicPDEDataT, str]="sinsinbi"):
        """
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('polyharmonic').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh: Union[Mesh, str] = "tri", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh(**kwargs)
        else:
            self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_space_degree(self, p: int):
        """
        Set the polynomial degree of the finite element space.

        Args:
            p: The polynomial degree.
        """
        self.p = p

    def set_smoothness(self, m: int):
        self.m = m

    def linear_system(self):
        """
        Construct the linear system for the eigenvalue problem.

        Returns:
            The stiffness matrix and mass matrix.
        """
        from ..functionspace import CrConformingFESpace2d
        from ..functionspace import CrConformingFESpace3d

        GD = self.mesh.geo_dimension()
        if self.mesh.TD == 2:
            self.space = CrConformingFESpace2d(self.mesh, self.p, self.m)
        if self.mesh.TD == 3:
            self.space = CrConformingFESpace3d(self.mesh, self.p, self.m)
        self.uh = self.space.function() # 建立一个有限元函数

        bform = BilinearForm(self.space)
        integrator = PolyharmonicIntegrator(m=self.m+1, coef=1, q=self.p+4)
        bform.add_integrator(integrator)

        lform = LinearForm(self.space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p+4))

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

    def solve(self):
        """
        Solve the linear system using a direct solver (default method).
        """
        from ..solver import spsolve
        A, F = self.linear_system()
        A, F = self.apply_bc(A, F)

        self.uh[:] =  spsolve(A, F, solver='scipy')
        self.logger.info(f"uh: {self.uh[:]}")

    def error(self):
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

        self.logger.info(f"l2: {l2}, h1: {h1}, h2: {h2}")



