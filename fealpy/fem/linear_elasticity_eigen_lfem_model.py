from typing import Any, Optional, Union
<<<<<<< HEAD
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
=======

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.model.linear_elasticity import LinearElasticityPDEDataT

from fealpy.mesh import Mesh
from fealpy.functionspace import functionspace 
from fealpy.fem import BilinearForm
from fealpy.fem import LinearElasticityIntegrator
from fealpy.fem import ScalarMassIntegrator as MassIntegrator

from fealpy.fem import DirichletBC

class LinearElasticityEigenLFEMModel(ComputationalModel):
    """
    A model for solving linear elasticity eigenvalue problems using the Lagrange finite element method (LFEM).

    This class constructs and solves a linear elasticity eigenvalue problem with arbitrary-order Lagrange finite elements.
    It encapsulates the configuration of the PDE model, mesh generation, material properties, and solver setup,
    providing a modular interface for flexible eigenvalue computations.

    Parameters:
        options (dict): A dictionary of configuration options used to initialize the model.
            Typically obtained from the :meth:`get_options` method.
            Includes keys such as 'pde', 'mesh_type', mesh division counts, and logging settings.

    Attributes:
        options (dict): The configuration options used to initialize the model.
        pde (object): The PDE data object selected according to ``options['pde']``.
        mesh (Mesh): The mesh object constructed based on ``options['mesh_type']`` and resolution parameters.
        space_degree (int): Polynomial degree of the Lagrange finite element space used for discretization.
        _private_attr (Any): Example of a private attribute (used internally).

    Methods:
        get_options(**kwargs): 
            Return a dictionary of default options, which can be overridden by keyword arguments.

    Examples:
>>>>>>> origin/develop
        >>> opts = LinearElasticityEigenLFEMModel.get_options(
        ...     mesh_type='uniform_tet',
        ...     nx=20, ny=20, nz=20,
        ...     pbar_log=False
        ... )
        >>> model = LinearElasticityEigenLFEMModel(opts)
        >>> model.run()
    """
<<<<<<< HEAD

    def __init__(self, options):

=======
    def __init__(self, options):
>>>>>>> origin/develop
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
<<<<<<< HEAD
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
=======
        """
        Generate a dictionary of default configuration options for the linear elasticity eigenvalue model.

        This method provides a convenient way to initialize the model's configuration.
        Users may call this method and override any subset of the default parameters to customize behavior,
        such as mesh resolution, finite element degree, PDE variant, and logging preferences.

        Parameters:
            pde (int, optional): Index of the linear elasticity PDE model to solve.
                Defaults to 1. The value selects from predefined model variants.

            mesh_type (str, optional): Type of the mesh used in the model.
                Defaults to 'uniform_tet'. Must be compatible with :meth:`set_init_mesh`.

            nx (int, optional): Number of subdivisions in the x-direction for mesh generation.
                Defaults to 10.

            ny (int, optional): Number of subdivisions in the y-direction.
                Defaults to 10.

            nz (int, optional): Number of subdivisions in the z-direction.
                Defaults to 10.

            space_degree (int, optional): Polynomial degree of the Lagrange finite element space.
                Defaults to 1.

            pbar_log (bool, optional): Whether to display a progress bar during execution.
                Defaults to True.

            log_level (str, optional): Logging verbosity level.
                Must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'. Defaults to 'INFO'.

        Returns:
            dict: A dictionary mapping configuration keys to their corresponding values.
                This dictionary is suitable for passing to the model constructor.

        Examples:
            >>> opts = LinearElasticityEigenLFEMModel.get_options(
            ...     mesh_type='uniform_quad',
            ...     nx=20, pbar_log=False
>>>>>>> origin/develop
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
<<<<<<< HEAD
=======
        TODO: update this interface to set_mesh
>>>>>>> origin/develop
        """
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](**kwargs)
        else:
            self.mesh = mesh
<<<<<<< HEAD
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
=======
        self.logger.info(self.mesh)
>>>>>>> origin/develop

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
        self.space = functionspace(self.mesh, ('Lagrange', self.p), shape=(-1, GD))

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
<<<<<<< HEAD
        self.bc.apply_matrix(S)
        self.bc.apply_matrix(M)
        return S.to_scipy(), M.to_scipy()

    def solve(self):
        """
        Solve the eigenvalue problem using the finite element method.

        Returns:
            Eigenvalues and eigenvectors of the system.
        """
=======
        isFreeDof = bm.logical_not(self.bc.is_boundary_dof)
        S = S.to_scipy()[isFreeDof, :][:, isFreeDof]
        M = M.to_scipy()[isFreeDof, :][:, isFreeDof]
        #self.bc.apply_matrix(S)
        #self.bc.apply_matrix(M)
        #return S.to_scipy(), M.to_scipy()
        return S, M

    @variantmethod('scipy')
    def solve(self, which: str = 'SM'):
        """Solve the eigenvalue problem using eigsh in scipy.

        Returns
            Eigenvalues and eigenvectors of the system.
        """
        from scipy.sparse.linalg import eigsh
>>>>>>> origin/develop
        S, M = self.linear_system()
        S, M = self.apply_bc(S, M)
        k = self.options.get('neign', 6)
        val, vec = eigsh(S, k=k, M=M, which='SM', tol=1e-6, maxiter=1000)

<<<<<<< HEAD
        self.show_modal(val, vec)

=======
        self.logger.info(f"Eigenvalues: {val}")

        self.show_modal(val, vec)

    @solve.register('slepc')
    def solve(self, which: str ='SM'):
        """Solve the eigenvalue problem using SLEPc.
        
        """
        from petsc4py import PETSc
        from slepc4py import SLEPc
        S, M = self.linear_system()
        S, M = self.apply_bc(S, M)

        S = PETSc.Mat().createAIJ(
                size=S.shape, 
                csr=(S.indptr, S.indices, S.data))
        S.assemble()
        M = PETSc.Mat().createAIJ(
                size=M.shape, 
                csr=(M.indptr, M.indices, M.data))
        M.assemble()

        eps = SLEPc.EPS().create()
        eps.setOperators(S, M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setTolerances(tol=1e-6, max_it=1000)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(1e-4)  # 目标 shift，通常为目标最小特征值附近

        ksp = st.getKSP()
        ksp.setType('cg')  # 或 'gmres'
        def my_ksp_monitor(ksp, its, rnorm):
            print(f"KSP iter {its}, residual norm = {rnorm}")
        ksp.setMonitor(my_ksp_monitor)
        pc = ksp.getPC()
        pc.setType('gamg')  # 或 'gamg' 若使用 AMG

        k = self.options.get('neign', 6)
        eps.setDimensions(nev=k, ncv=4*k)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(0.0)  # 目标特征值，通常为最小或最大特征值

        eps.setFromOptions()
        eps.solve()

        eigvals = []
        eigvecs = []

        vr, vi = eps.getOperators()[0].getVecs()
        print(f"Number of eigenvalues converged: {eps.getConverged()}")
        for i in range(min(k, eps.getConverged())):
            val = eps.getEigenpair(i, vr, vi)
            eigvals.append(val.real)
            eigvecs.append(vr.getArray().copy())
        val = bm.array(eigvals)
        vec = bm.stack(eigvecs, axis=1)
        self.logger.info(f"Eigenvalues: {val}")
        self.show_modal(val, vec)
>>>>>>> origin/develop

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

        GD = self.mesh.geo_dimension()
        node = self.mesh.entity('node')
        node0 = node.copy()
        isBdNode = self.pde.is_displacement_boundary(node)
<<<<<<< HEAD
=======
        isFreeNode = bm.logical_not(isBdNode)
>>>>>>> origin/develop

        fig = plt.figure()
        start = 231
        axes = fig.add_subplot(start, projection='3d')
        self.mesh.add_plot(axes)
        self.mesh.find_node(axes, index=isBdNode)

        for i in range(2, 7):
            start += 1
            u = vec[:, i - 2].reshape(-1, GD)
<<<<<<< HEAD
            node[:] += 0.01 * u
            print(u[isBdNode, :])
=======
            node[isFreeNode] += 0.02 * u
>>>>>>> origin/develop
            axes = fig.add_subplot(start, projection='3d')
            self.mesh.add_plot(axes)
            node[:] = node0
        plt.show()
<<<<<<< HEAD
           


=======
>>>>>>> origin/develop
