from typing import Any, Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod
<<<<<<< Updated upstream
from ..model import ComputationalModel, PDEDataManager
from ..model.linear_elasticity import LinearElasticityPDEDataT

from ..mesh import Mesh
from ..functionspace import functionspace 
from ..material import LinearElasticMaterial

=======
from ..model import ComputationalModel

from ..functionspace import functionspace
from ..material import LinearElasticMaterial
>>>>>>> Stashed changes
from ..fem import BilinearForm
from ..fem import LinearElasticityIntegrator
from ..fem import ScalarMassIntegrator as MassIntegrator

<<<<<<< Updated upstream
from ..fem import DirichletBC
from scipy.sparse.linalg import eigsh

=======
>>>>>>> Stashed changes
class LinearElasticityEigenLFEMModel(ComputationalModel):
    """
    A class to represent a linear elasticity eigenvalue problem using the
    Lagrange finite element method (LFEM).

    Attributes:
        mesh: The mesh of the domain.
    Reference:
        https://wnesm678i4.feishu.cn/wiki/HwBfwzraXi0ahYkwg72c7gMqn3e?fromScene=spaceOverview
    """

<<<<<<< Updated upstream
    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        self.set_material_parameters(self.pde.lam, self.pde.mu, self.pde.rho)
        self.set_init_mesh(options['init_mesh'])
        self.set_space_degree(options['space_degree'])


    def set_pde(self, pde: Union[LinearElasticityPDEDataT, str]="boxpoly3d"):
        """
        """
        if isinstance(pde, str):
            self.pde = PDEDataManager('linear_elasticity').get_example(pde)
        else:
            self.pde = pde 

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tet", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
=======
    def __init__(self, pbar_log=True, log_level="INFO"):
        super().__init__(pbar_log=pbar_log, log_level=log_level)


    def set_pde(self, pde):
        self.pde = pde

    def set_init_mesh(self, meshtype: str = "tri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)
>>>>>>> Stashed changes

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_material_parameters(self, lam: float, mu: float, rho: float):
<<<<<<< Updated upstream
        self.material = LinearElasticMaterial("eigens", lame_lambda=lam, shear_modulus=mu, density=rho)
=======
        self.material = LinearElasticMaterial(lame_lambda=lam, shear_modulus=mu, density=rho)
>>>>>>> Stashed changes

    def set_space_degree(self, p: int):
        """
        Set the polynomial degree of the finite element space.

        Args:
            p: The polynomial degree.
        """
        self.p = p

    def linear_system(self):
        """
        Construct the linear system for the eigenvalue problem.

        Returns:
            The stiffness matrix and mass matrix.
        """
        # Implementation of the linear system construction goes here
        GD = self.mesh.geo_dimension()
        space = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))

        bform = BilinearForm(space)
        integrator = LinearElasticityIntegrator(self.material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(space)
<<<<<<< Updated upstream
        integrator = MassIntegrator(self.material.density)
=======
        integrator = MassIntegrator(self.material.density())
>>>>>>> Stashed changes
        bform.add_integrator(integrator)
        M = bform.assembly()

        return S, M

<<<<<<< Updated upstream
    def apply_bc(self):
        """
        """
        pass

=======
>>>>>>> Stashed changes
    def solve(self):
        """
        Solve the eigenvalue problem using the finite element method.

        Returns:
            Eigenvalues and eigenvectors of the system.
        """
<<<<<<< Updated upstream
        S, M = self.linear_system()
        val, vec = eigsh(S.to_scipy(), k=6, M=M.to_scipy(), which='SM', tol=1e-5, maxiter=1000)
        self.logger.info(f"Eigenvalues: {val}")
=======
        # Implementation of the FEM solver goes here
        pass
>>>>>>> Stashed changes
