from typing import Any, Optional, Union
from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod
from ..model import ComputationalModel
from ..model.linear_elasticity import LinearElasticityPDEDataT

from ..mesh import Mesh
from ..functionspace import functionspace 
from ..material import LinearElasticMaterial

from ..fem import BilinearForm
from ..fem import LinearElasticityIntegrator
from ..fem import ScalarMassIntegrator as MassIntegrator

from ..fem import DirichletBC
from scipy.sparse.linalg import eigsh

class LinearElasticityEigenLFEMModel(ComputationalModel):
    """
    A class to represent a linear elasticity eigenvalue problem using the
    Lagrange finite element method (LFEM).

    Attributes
        mesh: The mesh of the domain.
    Reference
        https://wnesm678i4.feishu.cn/wiki/HwBfwzraXi0ahYkwg72c7gMqn3e?fromScene=spaceOverview
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        GD = self.pde.geo_dimension()

        self.set_material_parameters(self.pde.lam, self.pde.mu, self.pde.rho)
        mtype = options['mesh_type']
        if 'uniform' in mtype:
            if GD == 3:
                self.set_init_mesh(mtype, nx=options['nx'], ny=options['ny'], nz=options['nz'])
            elif GD == 2:
                self.set_init_mesh(mtype, nx=options['nx'], ny=options['ny'])
            else:
                raise ValueError(f"Unsupported mesh type {mtype} for geo_dimension {GD}.")
        else:
            self.set_init_mesh(mtype)
        self.set_space_degree(options['space_degree'])


    def set_pde(self, pde: Union[LinearElasticityPDEDataT, str]="boxdomain3d"):
        """
        """
        if isinstance(pde, str):
            self.pde = PDEModelManager('linear_elasticity').get_example(pde)
        else:
            self.pde = pde 

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tet", **kwargs):
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_material_parameters(self, lam: float, mu: float, rho: float):
        self.material = LinearElasticMaterial("eigens", lame_lambda=lam, shear_modulus=mu, density=rho)

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
        self.space = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))

        bform = BilinearForm(self.space)
        integrator = LinearElasticityIntegrator(self.material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(self.space)
        integrator = MassIntegrator(self.material.density)
        bform.add_integrator(integrator)
        M = bform.assembly()

        return S, M

    def apply_bc(self, S, M):
        """
        """
        from ..fem import DirichletBC
        
        bc = DirichletBC(
                self.space,
                gd=self.pde.displacement_bc,
                threshold=self.pde.is_displacement_boundary)
        S = bc.apply_matrix(S)
        M = bc.apply_matrix(M)
        return S, M

    def solve(self):
        """
        Solve the eigenvalue problem using the finite element method.

        Returns:
            Eigenvalues and eigenvectors of the system.
        """
        S, M = self.linear_system()
        val, vec = eigsh(S.to_scipy(), k=6, M=M.to_scipy(), which='SM', tol=1e-5, maxiter=1000)
        self.logger.info(f"Eigenvalues: {val}")

