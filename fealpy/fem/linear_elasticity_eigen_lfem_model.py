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
    """Model for linear elasticity eigenvalue problems using the LFEM.

    This class sets up and solves a linear elasticity eigenvalue problem
    with arbitrary‐order Lagrange finite elements.  It encapsulates PDE
    selection, mesh initialization, material parameters, and solver settings.

    Parameters
        options : dict
            Configuration options for the model, as returned by :meth:`get_options`.

    Attributes
        options : dict
            The configuration options passed to the model.
        pde : object
            The PDE data instance selected based on ``options['pde']``.
        mesh : Mesh
            The mesh object initialized according to ``options['mesh_type']`` and division counts.
        space_degree : int
            Polynomial degree of the finite element space.
        _private_attr : any
            (Optional) Example of a private attribute.

    Methods
        get_options(...)
            Return a dict of default options, which users may override.
    Examples
        >>> opts = LinearElasticityEigenLFEMModel.get_options(
        ...     mesh_type='uniform_tet',
        ...     nx=20, ny=20, nz=20,
        ...     pbar_log=False
        ... )
        >>> model = LinearElasticityEigenLFEMModel(opts)
        >>> model.run()
    """

    def __init__(self, options):

        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )
        self.set_pde(options['pde'])
        GD = self.pde.geo_dimension()

        mtype = options['mesh_type']
        if 'uniform' in mtype:
            if GD == 3:
                self.set_init_mesh(
                    mtype,
                    nx=options['nx'],
                    ny=options['ny'],
                    nz=options['nz']
                )
            elif GD == 2:
                self.set_init_mesh(
                    mtype,
                    nx=options['nx'],
                    ny=options['ny']
                )
            else:
                raise ValueError(
                    f"Unsupported mesh type {mtype} for geo_dimension {GD}."
                )
        else:
            self.set_init_mesh(mtype)

        self.set_space_degree(options['space_degree'])

    @classmethod
    def get_options(
        cls,
        pde: int = 1,
        mesh_type: str = 'uniform_tet',
        nx: int = 10,
        ny: int = 10,
        nz: int = 10,
        space_degree: int = 1,
        pbar_log: bool = True,
        log_level: str = 'INFO',
    ) -> dict:
        """Generate a dict of default configuration options for the model.

        Users may call this method and override any subset of parameters
        to customize model behavior.

        Parameters
            pde : int, optional, default=1
                Index of the linear elasticity PDE model to solve.
            mesh_type : str, optional, default='uniform_tet'
                Mesh type identifier used in :meth:`set_init_mesh`.
            nx : int, optional, default=10
                Number of divisions along the x‐direction for uniform meshes.
            ny : int, optional, default=10
                Number of divisions along the y‐direction for uniform meshes.
            nz : int, optional, default=10
                Number of divisions along the z‐direction for uniform meshes.
            space_degree : int, optional, default=1
                Degree of the Lagrange finite element space.
            pbar_log : bool, optional, default=True
                Whether to display a progress bar.
            log_level : str, optional, default='INFO'
                Logging level; one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        Returns
            options : dict
                A dictionary mapping each option name to its value.

        Examples
            >>> opts = LinearElasticityEigenLFEMModel.get_options(
            ...     mesh_type='uniform_quad', nx=20, pbar_log=False
            ... )
            >>> print(opts['mesh_type'], opts['nx'], opts['pbar_log'])
            uniform_quad 20 False
        """
        return {
            'pde': pde,
            'mesh_type': mesh_type,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'space_degree': space_degree,
            'pbar_log': pbar_log,
            'log_level': log_level,
        }

    def set_init_mesh(self, mesh: Union[Mesh, str] = "uniform_tet", **kwargs):
        """
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_pde(self, pde: Union[LinearElasticityPDEDataT, int] = 1) -> None:
        if isinstance(pde, int):
            self.pde = PDEModelManager("linear_elasticity").get_example(1)
        else:
            self.pde = pde
        self.logger.info(self.pde)
        self.logger.info(self.pde.material)

    def set_space_degree(self, p: int):
        """
        """
        self.p = p

    def linear_system(self):
        """
        """
        # Implementation of the linear system construction goes here
        GD = self.mesh.geo_dimension()
        self.space = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))

        bform = BilinearForm(self.space)
        integrator = LinearElasticityIntegrator(self.pde.material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(self.space)
        integrator = MassIntegrator(self.pde.material.density)
        bform.add_integrator(integrator)
        M = bform.assembly()

        return S, M

    def apply_bc(self, S, M):
        """
        """
        from ..fem import DirichletBC
        
        self.bc = DirichletBC(
                self.space,
                gd=self.pde.displacement_bc,
                threshold=self.pde.is_displacement_boundary)
        self.bc.apply_matrix(S)
        self.bc.apply_matrix(M)
        return S.to_scipy(), M.to_scipy()

    def solve(self):
        """
        Solve the eigenvalue problem using the finite element method.

        Returns:
            Eigenvalues and eigenvectors of the system.
        """
        S, M = self.linear_system()
        S, M = self.apply_bc(S, M)
        k = self.options.get('neign', 6)
        val, vec = eigsh(S, k=k, M=M, which='SM', tol=1e-6, maxiter=1000)


    def show_mesh(self):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        node = self.mesh.entity('node')
        isBdNode = self.pde.is_displacement_boundary(node)
        self.mesh.add_plot(axes)
        self.mesh.find_node(axes, index=isBdNode)
        plt.show()


    def show_modal(self, val, vec):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D


