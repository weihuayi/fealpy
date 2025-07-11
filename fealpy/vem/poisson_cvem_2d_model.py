from typing import Any, Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod
from ..model import ComputationalModel, PDEDataManager
from ..model.poisson import PoissonPDEDataT

from ..mesh import Mesh, PolygonMesh
from ..functionspace import functionspace 
from ..vem import BilinearForm                              
from ..functionspace import ConformingScalarVESpace2d                         
from ..vem import LinearForm                                  
from ..vem import ScalarDiffusionIntegrator
from ..vem import ScalarSourceIntegrator         
from ..vem import DirichletBC                                

class PoissonCVEMModel(ComputationalModel):
    """
    A class representing the conforming virtual element method (CVEM)
    for solving the 2D Poisson equation.

    Attributes:
        pde: The PDE problem with source and exact solution.
        mesh: The polygonal mesh.
        p: Polynomial degree.
        space: The virtual element space.
        A: Stiffness matrix.
        F: Load vector.
        uh: Numerical solution.
    """

    def __init__(self, options):
        """
        Initialize the CVEM model from a configuration dictionary.

        Args:
            options (dict): Configuration options including:
                - 'pde' (str or PoissonPDEDataT): PDE specification.
                - 'mesh_type' (str): Type of polygonal mesh (e.g., 'uniform_poly').
                - 'nx', 'ny' (int): Mesh resolution parameters.
                - 'space_degree' (int): Polynomial degree of the virtual element space.
        """
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        GD = self.pde.geo_dimension()
        self.set_init_mesh(options['mesh_type'], nx=options['nx'], ny=options['ny'])
        self.set_space_degree(options['space_degree'])

    def set_pde(self, pde: Union[PoissonPDEDataT, str]="sinsin"):
        """
        Set the PDE data.

        Args:
            pde (Union[str, PoissonPDEDataT]): Either a predefined example key,
            or a user-defined PoissonPDEDataT instance.
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('poisson').get_example(pde)
        else:
            self.pde = pde 

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_poly", **kwargs):
        """
        Initialize the polygonal mesh.
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = PolygonMesh.from_mesh(mesh)

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


    def linear_system(self):
        """
        Assemble the CVEM linear system (stiffness matrix and load vector).

        Returns:
            (A, F): Assembled stiffness matrix and right-hand-side vector.
        """
        self.space = ConformingScalarVESpace2d(self.mesh, p=self.p)
        self.uh = self.space.function() # 建立一个有限元函数

        bform = BilinearForm(self.space)
        diff = ScalarDiffusionIntegrator(coef=1, q=self.p + 3)
        bform.add_integrator(diff)
        A = bform.assembly()

        lform = LinearForm(self.space)
        src = ScalarSourceIntegrator(self.pde.source, q=self.p + 3)
        lform.add_integrator(src)
        F = lform.assembly()

        bc = DirichletBC(self.space, self.pde.dirichlet)
        A, F = bc.apply(A, F)
        return A, F

    def apply_bc(self, A, F):
        """
        Apply Dirichlet boundary conditions.

        Returns:
            A, F: Modified matrix and vector after applying boundary conditions.
        """

        from ..fem import DirichletBC
        A, F = DirichletBC( self.space, gd=self.pde.dirichlet).apply(A, F)
        return A, F

    def solve(self):
        """
        Solve the linear system using a direct solver (default method).
        """
        from ..solver import spsolve
        A, F = self.linear_system()
        A, F = self.apply_bc(A, F)

        self.uh[:] =  spsolve(A, F, solver='scipy')

        self.sh = self.space.project_to_smspace(self.uh)
        self.logger.info(f"uh: {self.uh[:]}")

    def error(self):
        """
        Compute L2, H1 errors of the numerical solution.

        Returns:
            l2: L2-norm error between exact and numerical solution.
            h1: H1-seminorm error.
        """
        from ..decorator import barycentric

        l2 = self.mesh.error(self.sh.value, self.pde.solution)
        h1 = self.mesh.error(self.sh.grad_value, self.pde.gradient)

        self.logger.info(f"l2: {l2}, h1: {h1}")



