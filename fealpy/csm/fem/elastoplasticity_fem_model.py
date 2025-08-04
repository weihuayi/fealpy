
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import functionspace
from fealpy.material import LinearElasticMaterial

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import ScalarNeumannBCIntegrator, ScalarSourceIntegrator

from ..model import CSMModelManager
from ..material import ElastoplasticMaterial

from . import ElastoplasticitySourceIntIntegrator
from . import ElastoplasticDiffusionIntegrator


class ElastoplasticityFEMModel(ComputationalModel):
    """
    ElastoplasticityFEMModel is a finite element model for solving elastoplasticity problems.

    This class implements a finite element method (FEM) solver for elastoplasticity problems, 
    suitable for analyzing materials that exhibit both elastic and plastic behavior under loading. 
    By specifying material parameters, boundary conditions, and loads, it automatically constructs 
    the finite element mesh, assembles the stiffness matrix and load vector, and solves for the displacement field. 
    It relies on PDEModelManager for physical parameters and boundary conditions, making it suitable for teaching, 
    research, and preliminary engineering analysis.
    
    Parameters:
        example : str, optional, default='elastoplasticity2d'
            Selects a preset elastoplasticity problem example for initializing PDE parameters and mesh.
            
    Attributes:
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
            
    Methods:
        run()
            Executes the FEM solution process and returns the displacement vector.
        linear_system()
            Assembles and returns the stiffness matrix and load vector for the elastoplasticity problem.
        solve()
            Applies boundary conditions and solves the linear system, returning the displacement solution.
    """
    def __init__(self, options):
        '''
        Initializes the ElastoplasticityFEMModel with the specified example.
        
        Parameters:
            example (str): The name of the elastoplasticity problem example to use. Default is 'elastoplasticity2d'.
            Initializes the PDE parameters, mesh, and material properties based on the example.
            
        Raises:
            ValueError: If the example is not recognized or cannot be initialized.
        '''
        self.options = options
        super().__init__(pbar_log=options['pbar_log'], log_level=options['log_level'])
        self.set_pde(options['pde'])
        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)
        self.set_space_degree(options['space_degree'])
        self.set_space()
        self.set_material_parameters(E=2.069e5, nu=0.29)  # Set material parameters
        self.E = self.pde.E
        self.nu = self.pde.nu
        self.f = self.pde.Ft_max
        self.N = 20  # Number of load steps


    def set_pde(self, pde=1) -> None:
        '''
        Set the PDE parameters for the elastoplasticity problem.
        '''
        if isinstance(pde, int):
            self.pde = CSMModelManager('elastoplasticity').get_example(pde)
        else:
            self.pde = pde
        self.logger.info(self.pde)
            
    def set_space_degree(self, p: int = 1):
        '''
        Set the polynomial degree for the finite element space.
        '''
        self.p = p

    def set_mesh(self, mesh: Mesh) -> Mesh:
        '''
        Initialize the finite element mesh based on the PDE parameters.
        This method generates the mesh according to the domain and boundary conditions defined in the PDE data.
        
        Parameters:
            mesh (Mesh): The mesh object to be initialized.

        Returns:
            Mesh: The initialized mesh object.
        '''
        self.mesh = mesh
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
        
    def set_space(self):
        """
        Set the finite element space for the model based on the mesh and polynomial degree.
        """
        GD = self.mesh.geo_dimension()
        self.space = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))
        Ldof = self.space.number_of_local_dofs()
        GDof = self.space.number_of_global_dofs()
        self.logger.info(f"Lagrange space: {self.space}, LDOF: {Ldof}, GDOF: {GDof}")

    def set_material_parameters(self, E: float, nu: float):
        """
        Set the material parameters for the elastoplasticity model.
        
        Parameters:
            E (float): Young's modulus in MPa.
            nu (float): Poisson's ratio.
        """
        self.cm = LinearElasticMaterial("elastic", elastic_modulus=E, poisson_ratio=nu, 
                                              hypo='plane_stress', device=bm.get_device(self.mesh))
        qf = self.mesh.quadrature_formula(q=self.space.scalar_space.p+3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        self.NQ = bcs.shape[0]
        self.NC = self.mesh.number_of_cells()
        self.B = self.cm.strain_matrix(True, gphi=self.space.scalar_space.grad_basis(bcs))
        self.D = self.cm.elastic_matrix()
        self.pfcm = ElastoplasticMaterial(name='elastoplastic',
                                elastic_modulus=E, poisson_ratio=nu,
                                yield_stress=self.pde.yield_stress, hardening_modulus=self.pde.hardening_modulus, hypo='plane_stress', device=bm.get_device(self.mesh))

    def cell2dof(self, stress):
        """
        Get the mapping from cell indices to global degree of freedom (dof) numbers.

        Parameters:
            stress: Stress tensor for the cells.    
            
        Returns:
            TensorLike: A tensor containing the global dof numbers for each cell.
        """
        stress = stress[self.space.cell_to_dof()]
        return stress
    
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

        # Compute stress
        stress = bm.einsum('cqij,cqj->cqi', self.D, strain_elastic)
        return stress
    
    def compute_strain(self, displacement):
        """
        Compute the strain field based on the current displacement field.
        """
        node = self.mesh.entity('node')
        kwargs = bm.context(node)
        cell2dof = self.space.cell_to_dof()
        uh = bm.array(displacement, **kwargs)
        uh_cell = uh[cell2dof]
        strain = bm.einsum('cqij,cj->cqi', self.B, uh_cell)
        return strain

    def linear_system(self, source, D_ep, stress):
        '''
        Construct the linear system for the elastoplasticity problem.
        
        Parameters:
            source (callable): Function defining the external load vector.
            D_ep (TensorLike): Elastoplastic material stiffness matrix.
            stress (TensorLike): Stress tensor for the cells.

        Returns:
            Tuple[TensorLike, TensorLike, TensorLike]:
                - K (TensorLike): Assembled stiffness matrix.
                - F_int (TensorLike): Internal force vector.
                - F_ext (TensorLike): External force vector.    
        '''
        elastoplasticintegrator= ElastoplasticDiffusionIntegrator(D_ep, material=self.pfcm,
                                    q=self.space.p+3)
        bform = BilinearForm(self.space)
        bform.add_integrator(elastoplasticintegrator)
        K = bform.assembly(format='csr')

        lform = LinearForm(self.space) 
        lform.add_integrator(ScalarSourceIntegrator(source=source))
        lform.add_integrator(ScalarNeumannBCIntegrator(
            source=self.pde.neumann,
            threshold=self.pde.neumann_boundary,
            q=self.space.scalar_space.p + 3
        ))
        F_ext = lform.assembly()
        
        lform2 = LinearForm(self.space)
        elastoplasticintintegrator = ElastoplasticitySourceIntIntegrator(strain_matrix=self.B, stress=stress,
            q=self.space.p + 3,
        )
        lform2.add_integrator(elastoplasticintintegrator)
        F_int = lform2.assembly()

        return K , F_int, F_ext
      
    
    def solve(self):
        # 初始化
        u = self.space.function()  # 初始位移
        strain_pl = bm.zeros((self.NC, self.NQ, 3), dtype=bm.float64)
        stress = bm.zeros((self.NC, self.NQ, 3), dtype=bm.float64)  # 初始应力
        strain_e = bm.zeros((self.NC, self.NQ), dtype=bm.float64)  # 初始等效塑性应变
        tol = 1e-8
        D_ep = self.pfcm.elastic_matrix()  # 弹塑性矩阵
        N = self.N

        # 循环荷载步
        for n in range(N):
            delta_u = 0
            converged = False
            print(f"Load step {N}, solving...")
            while not converged:
                from fealpy.decorator import cartesian
                @cartesian
                def loading(p):
                    coef = (n + 1) / N
                    val = coef * self.pde.source(p)
                    return val
                K, F_int, F_ext = self.linear_system(source=loading, D_ep=D_ep, stress=stress)
                R = F_int - F_ext
                print(F_int.max(), F_ext.max()) #F_int一直在变大
                if bm.linalg.norm(R) > 10e6:
                    exit()
                from fealpy.solver import spsolve
                delta_u = spsolve(K, -R,'scipy')

                # 更新位移增量 Δu ← Δu + δΔu
                delta_u += delta_u
                
                delta_strain = self.compute_strain(delta_u)

                # 更新所有积分点状态
                stress, strain_pl, strain_e, D_ep, is_plastic = self.pfcm.material_point_update(delta_strain, strain_pl, strain_e)
                strain_e += strain_e

                # 检查残差和位移增量是否满足收敛条件
                if bm.linalg.norm(R) < tol and bm.linalg.norm(delta_u) < tol:
                    converged = True

            # 更新总位移
            u += delta_u
            
            
        # 返回最终结果或状态
        return u, stress, strain_pl, strain_e

    def save_data(self, filename):
        pass

       
