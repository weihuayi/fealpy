from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from ..model import CSMModelManager
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import Mesh
from fealpy.solver import spsolve
from typing import Union
from ..model.elastoplasticity import ElastoplasticityPDEDataT
from fealpy.fem import VectorSourceIntegrator
from fealpy.material import LinearElasticMaterial
from ..material import ElastoplasticMaterial
from ...functionspace import functionspace

class ElastoplasticityFEMModel(ComputationalModel):
    """
    ElastoplasticityFEMModel is a finite element model for solving elastoplasticity problems.

    This class implements a finite element method (FEM) solver for elastoplasticity problems, 
    suitable for analyzing materials that exhibit both elastic and plastic behavior under loading. 
    By specifying material parameters, boundary conditions, and loads, it automatically constructs 
    the finite element mesh, assembles the stiffness matrix and load vector, and solves for the displacement field. 
    It relies on PDEModelManager for physical parameters and boundary conditions, making it suitable for teaching, 
    research, and preliminary engineering analysis.
    
    Parameters
        example : str, optional, default='elastoplasticity2d'
            Selects a preset elastoplasticity problem example for initializing PDE parameters and mesh.
            
    Attributes
        pde : object
            Object returned by PDEModelManager containing physical parameters and boundary conditions.
        mesh : object
            Finite element mesh object describing the discretized domain.
        E : float
            Young's modulus, describing material stiffness.
        nu : float
            Poisson's ratio, describing material compressibility.
        yield_strength : float
            Yield strength of the material.
        f : float or callable
            Distributed load applied to the domain.
        l : float
            Characteristic length of the domain.
            
    Methods
        run()
            Executes the FEM solution process and returns the displacement vector.
        linear_system()
            Assembles and returns the stiffness matrix and load vector for the elastoplasticity problem.
        solve()
            Applies boundary conditions and solves the linear system, returning the displacement solution.
            
    Notes
        This class assumes the provided PDEModelManager example defines all necessary parameters and boundary conditions.
        Supports custom loads and boundary conditions for various elastoplasticity problems.
        Depends on external finite element spaces, integrators, and linear solvers.
    """
    def __init__(self, options):
        '''
        Initializes the ElastoplasticityFEMModel with the specified example.
        Parameters
            example (str): The name of the elastoplasticity problem example to use. Default is 'elastoplasticity2d'.
            Initializes the PDE parameters, mesh, and material properties based on the example.
        Raises
            ValueError: If the example is not recognized or cannot be initialized.
        Notes
            The example should be a valid key in the PDEModelManager for elastoplasticity problems.
            It must define all necessary parameters and boundary conditions.
        '''
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)
        self.E = self.pde.E
        self.nu = self.pde.nu
        self.yield_strength = self.pde.sigma_y0
        self.f = self.pde.Ft_max
        self.a = self.pde.a
        self.N = 10


    def set_pde(self, pde:Union[ElastoplasticityPDEDataT, int]=1):
        '''
        Set the PDE parameters for the elastoplasticity problem.
        Parameters
            pde (PDEModelManager): The PDE data manager containing elastoplasticity parameters and boundary conditions.
        Raises
            ValueError: If the provided pde is not valid or does not contain necessary parameters.
        Notes
            This method updates the model's physical parameters and mesh based on the provided PDE data.
        '''
        if isinstance(pde, int):
            self.pde = CSMModelManager('elastoplasticity').get_example(pde)
        else:
            self.pde = pde
        self.logger.info(self.pde)

    def set_mesh(self, mesh: Mesh):
        '''
        This method generates the mesh according to the domain and boundary conditions defined in the PDE data.
        
        Parameters
            mesh (Mesh): The finite element mesh object to be used in the model.
        Raises
            ValueError: If the mesh cannot be initialized due to invalid parameters.
        Notes
            The mesh is based on the domain and boundary conditions defined in the PDE data.
        This method initializes the mesh and logs its properties.
        Returns 
            None
        '''
        self.mesh = mesh

        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")

    def set_material_parameters(self, lam: float, mu: float):
        self.material = LinearElasticMaterial("linear_elastic", lame_lambda=lam, shear_modulus=mu,
                                              hypo='plane_stress', device=bm.get_device(self.mesh))
        qf = self.mesh.quadrature_formula(q=self.mesh.scalar_space.p+3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        self.B = self.material.strain_matrix(True, gphi=self.mesh.scalar_space.grad_basis(bcs))
        self.D = self.material.elastic_matrix()
        self.pfcm = ElastoplasticMaterial(name='elastoplastic',
                                elastic_modulus=1e5, poisson_ratio=0.3,
                                yield_stress=50, hardening_modulus=0.0, hypo='plane_stress', device=bm.get_device(self.mesh))
        self.logger.info(self.pfcm)

    def set_space(self):
        '''
        This method initializes the function space used for the finite element discretization.
        '''
        GD = self.mesh.geo_dimension()
        self.space = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))

    # Compute stress based on the current displacement field
    def compute_stress(self, displacement, plastic_strain):
        """
        Compute the stress field based on the current displacement field.
        """
        node = self.mesh.entity('node')
        kwargs = bm.context(node)
        cell2dof = self.space.cell_to_dof()
        uh = bm.array(displacement,**kwargs)
        uh_cell = uh[cell2dof]
        strain_total = bm.einsum('cqij,cj->cqi', self.B, uh_cell)
        strain_elastic = strain_total - plastic_strain

        # 计算应力
        stress = bm.einsum('cqij,cqj->cqi', self.D, strain_elastic)
        return stress
    
    # Assemble internal force term based on current stress
    def compute_internal_force(self, stress):
        qf = self.mesh.quadrature_formula(q=self.mesh.scalar_space.p+3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        cm = self.mesh.entity_measure('cell')
        F_int_cell = bm.einsum('q, c, cqij,cqi->cj', 
                             ws, cm, self.B, stress) # (NC, tdof)
        
        return F_int_cell

    def linear_system(self):
        '''
        Assemble the linear system for the elastoplasticity problem.
        This method constructs the stiffness matrix and load vector based on the finite element space and PDE parameters.
        Returns
            tuple: (A, b) where A is the stiffness matrix and b is the load vector.
        Raises
            ValueError: If the assembly fails due to invalid parameters or mesh.
        Notes
            The method uses bilinear forms and linear forms defined in the FEM library.
        '''
        node = self.mesh.entity('node')
        kwargs = bm.context(node)
        NC = self.mesh.number_of_cells()
        NQ = self.mesh.number_of_nodes()
        equivalent_plastic_strain = bm.zeros((NC, NQ),**kwargs)
        self.pfcm = ElastoplasticMaterial(name='E1nu0.3',
                                elastic_modulus=1e5, poisson_ratio=0.3,
                                yield_stress=50, hardening_modulus=0.0, hypo='plane_stress', device=bm.get_device(self.mesh))
        from . import ElastoplasticIntegrator
        elasticintegrator= ElastoplasticIntegrator(self.pfcm.D_ep, material=self.pfcm,space=tensor_space, 
                                    q=tensor_space.p+3)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(elasticintegrator)
        K = bform.assembly(format='csr')


        load = self.f
        space = LagrangeFESpace(self.mesh, p=1, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

        lform = LinearForm(tensor_space) 
        lform.add_integrator(VectorSourceIntegrator(source = load))
        F_ext = lform.assembly()

        elasticintegrator = ElastoplasticIntegrator(
            material=self.pde.material,
            space=tensor_space,
            q=tensor_space.p + 3,
            equivalent_plastic_strain=self.pde.equivalent_plastic_strain
        )

        F_int = elasticintegrator.assembly()
        return K , F_int, F_ext
    
    def solve(self):
        # 初始化
        u = 0  # 初始位移
        sigma = 0  # 初始应力
        alpha = 0  # 初始塑性应变
        ep = 0  # 初始等效塑性应变
        tol = 1e-8
        N = self.N

        # 循环荷载步
        for n in range(N):
            delta_u = 0
            converged = False
            while not converged:
                K, F_int, F_ext = self.linear_system()
                R = F_int - F_ext
                from fealpy.solver import spsolve
                delta_u = spsolve(K, -R)
                delta_u += delta_u
                sigma, alpha, ep = self.plasticity_update_2d(u, sigma, alpha, ep, delta_u)

                if bm.norm(R) < tol and bm.norm(delta_u) < tol:
                    converged = True
            u += delta_u
            self.save_state(sigma, alpha, ep)

        # 返回最终结果或状态
        return u, sigma, alpha, ep

       
