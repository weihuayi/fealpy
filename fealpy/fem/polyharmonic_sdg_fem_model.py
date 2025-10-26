from typing import Optional, Union
from fealpy.backend import bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.model.polyharmonic import PolyharmonicPDEDataT
from fealpy.decorator import variantmethod

from fealpy.mesh import Mesh
from fealpy.functionspace import functionspace

from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC


class PolyharmonicSDGFEMModel(ComputationalModel):
    """
    Solves the high-order PDE \(\Delta^{m+1} u = f\) on a 2D or 3D domain 
    using Primal Staggered Discontinuous Galerkin method.

    Attributes:
        mesh (Mesh): The computational mesh used for discretization.
            Defines the geometric domain and its connectivity.

        pde (PDE): The problem specification, including source terms,
            boundary conditions, and exact solutions (if available).

        p (int): The polynomial degree of the finite element space.
            Determines the approximation order for the solution.

        m (int): The smoothness order of the finite element space.

    """
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_init_mesh(options['init_mesh'], nx=options['mesh_size'], ny=options['mesh_size'] )
        self.set_space_degree(options['space_degree'])
        self.set_smoothness(options['smoothness'])

    def set_pde(self, pde: Union[PolyharmonicPDEDataT, int]=1):
        """
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('polyharmonic').get_example(pde)
        else:
            self.pde = pde

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_poly", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
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
            p(int): The polynomial degree.
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
        pass

    def apply_bc(self, A, F):
        """
        Apply Dirichlet boundary conditions.

        """

        from ..fem import DirichletBC
        pass

    def solve(self):
        """
        Solve the linear system using a direct solver (default method).
        """
        from ..solver import spsolve
        pass

    def error(self):
        """
        Compute L2, H1, and H2 errors of the numerical solution.

        Returns:
            l2: L2-norm error between exact and numerical solution.
            h1: H1-seminorm error.
            h2: H2-seminorm error (approximate if pde.hessian is provided).
        """
        from ..decorator import barycentric
        pass

       

