from typing import Any, Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod
from ..model import ComputationalModel

from ..functionspace import function_space 
from ..material import LinearElasticMaterial
from ..fem import BilinearForm
from ..fem import LinearElasticityIntegrator
from ..fem import ScalarMassIntegrator as MassIntegrator

class LinearElasticityEigenLFEMModel(ComputationalModel):
    """
    A class to represent a linear elasticity eigenvalue problem using the
    Lagrange finite element method (LFEM).
    
    Attributes:
        mesh: The mesh of the domain.
    Reference:
        https://wnesm678i4.feishu.cn/wiki/HwBfwzraXi0ahYkwg72c7gMqn3e?fromScene=spaceOverview
    """

    def __init__(self, pbar_log=True, log_level="INFO"):
        super().__init__(pbar_log=pbar_log, log_level=log_level)


    def set_pde(self, pde):
        self.pde = pde 

    def set_init_mesh(self, meshtype: str = "tri", **kwargs):
        self.mesh = self.pde.init_mesh[meshtype](**kwargs)

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_material_parameters(self, lam: float, mu: float, rho: float):
        self.material = LinearElasticMaterial(lame_lambda=lam, shear_modulus=mu, density=rho)

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
        space = function_space(self.mesh, ('Lagrange', self.p), shape=(GD, -1))

        bform = BilinearForm(space)
        integrator = LinearElasticityIntegrator(self.material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(space)
        integrator = MassIntegrator(self.material.density())
        bform.add_integrator(integrator)
        M = bform.assembly()

        return S, M

    def solve(self):
        """
        Solve the eigenvalue problem using the finite element method.
        
        Returns:
            Eigenvalues and eigenvectors of the system.
        """
        # Implementation of the FEM solver goes here
        pass
