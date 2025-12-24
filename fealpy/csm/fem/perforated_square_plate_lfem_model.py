
from typing import Union

from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.mesher import BlockWithHoleMesher
from fealpy.functionspace import functionspace
from fealpy.material import LinearElasticMaterial

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import ScalarNeumannBCIntegrator

from ..model import CSMModelManager
from ..material import ElastoplasticMaterial
from ..utils import timer

from . import ElastoplasticitySourceIntIntegrator
from . import ElastoplasticDiffusionIntegrator


class PerforatedSquarePlateFEMModel(ComputationalModel):
    """
    PerforatedSquarePlateFEMModel is a finite element model for solving elastoplasticity problems.

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
        self.set_init_mesh()
        self.set_space_degree(options['space_degree'])
        self.set_space()
        self.set_material_parameters(E=10000, nu=0.25)  # Set material parameters
        self.E = self.pde.E
        self.nu = self.pde.nu
        self.f = self.pde.Ft_max
        self.N = 20  # Number of load steps


    def set_pde(self, pde=4) -> None:
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
    """
    def set_init_mesh(self, mesh: Union[Mesh, str] = "tri_threshold", **kwargs ):
        '''
        This method generates the mesh according to the domain and boundary conditions defined in the PDE data.
        
        Parameters:
            mesh (Mesh): The mesh object to be initialized.

        Returns:
            Mesh: The initialized mesh object.
        '''
        if isinstance(mesh, str):
            self.mesh = self.pde.init_mesh[mesh](big_box=(-5,5,-5,5), small_box=(-5,0,-5,0), nx=10, ny=10)
        else:
            self.mesh = mesh
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        NF = self.mesh.number_of_faces()
        NC = self.mesh.number_of_cells()
        self.logger.info(f"Mesh initialized with {NN} nodes, {NE} edges, {NF} faces, and {NC} cells.")
    """
    def set_init_mesh(self):
        '''
        This method generates the mesh according to the domain and boundary conditions defined in the PDE data.
        
        Parameters:
            mesh (Mesh): The mesh object to be initialized.

        Returns:
            Mesh: The initialized mesh object.
        '''
        mesher = BlockWithHoleMesher()
        self.mesh = mesher.mesh
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
                                              hypo='3D', device=bm.get_device(self.mesh))
        qf = self.mesh.quadrature_formula(q=self.space.scalar_space.p+3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        self.NQ = bcs.shape[0]
        self.NC = self.mesh.number_of_cells()
        self.B = self.cm.strain_matrix(True, gphi=self.space.scalar_space.grad_basis(bcs))
        self.D = self.cm.elastic_matrix()
        self.pfcm = ElastoplasticMaterial(name='elastoplastic',
                                elastic_modulus=E, poisson_ratio=nu,
                                yield_stress=self.pde.yield_stress, 
                                hardening_modulus=self.pde.hardening_modulus, 
                                hypo='3D', device=bm.get_device(self.mesh))

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

    def linear_system(self, loading_neumann, D_ep, stress):
        '''
        Construct the linear system for the elastoplasticity problem.
        
        Parameters:
            loading_neumann (callable): Function defining the Neumann boundary load vector.
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
        KK = elastoplasticintegrator.assembly(self.space)
        bform = BilinearForm(self.space)
        bform.add_integrator(elastoplasticintegrator)
        K = bform.assembly(format='csr')
        lform2 = LinearForm(self.space)
        lform2.add_integrator(ScalarNeumannBCIntegrator(
            source=loading_neumann,
            threshold=self.pde.neumann_boundary,
            q=self.space.scalar_space.p + 3
        ))
        F_ext = lform2.assembly()
        lform2 = LinearForm(self.space)
        elastoplasticintintegrator = ElastoplasticitySourceIntIntegrator(strain_matrix=self.B, stress=stress,
            q=self.space.p + 3,
        )
        lform2.add_integrator(elastoplasticintintegrator)
        F_int = lform2.assembly()

        return K , F_int, F_ext
       
    def solve(self):

        tmr = timer()
        next(tmr)

        u = self.space.function()  # 总位移
        strain_pl = bm.zeros((self.NC, self.NQ, 6), dtype=bm.float64)
        stress = bm.zeros((self.NC, self.NQ, 6), dtype=bm.float64)
        strain_e = bm.zeros((self.NC, self.NQ), dtype=bm.float64)
        strain_total = bm.zeros((self.NC, self.NQ, 6), dtype=bm.float64)

        tol = 1e-6
        N = self.N

        for n in range(N):
            delta_u = bm.zeros_like(u.array)
            converged = False
            print(f"Load step {n+1}/{N}, solving...")

            # 固定的加载项
            from fealpy.decorator import cartesian

            @cartesian
            def loading_neumann(p):
                coef = (n + 1) / N
                return coef * self.pde.neumann(p)

            # 记录状态
            strain_pl_old = bm.copy(strain_pl)
            strain_e_old = bm.copy(strain_e)

            iter_count = 0
            while not converged:
                # 计算当前试应变
                delta_strain = self.compute_strain(delta_u)
                print(strain_total.shape, delta_strain.shape)
                strain_total_trial = strain_total + delta_strain
                tmr.send(f'第{n}次迭代：试应变计算时间')

                stress_trial, strain_pl_trial, strain_e_trial, Ctang_trial, is_plastic = \
                    self.pfcm.material_point_update(strain_total_trial, strain_pl_old, strain_e_old)
                tmr.send(f'第{n}次迭代：材料点更新计算时间')
                    
                #num_plastic = is_plastic.astype(bm.uint8).sum()
                num_plastic = bm.sum(bm.astype(is_plastic, bm.uint8))
                num_total = is_plastic.size

                if num_plastic == 0:
                    print("====== 当前为纯弹性阶段 ======")
                else:
                    print(f"====== 当前为塑性阶段：{num_plastic}/{num_total} 个点屈服 ======")

                K, F_int, F_ext = self.linear_system(loading_neumann=loading_neumann,
                                                    D_ep=Ctang_trial,
                                                    stress=stress_trial)  
                tmr.send(f'第{n}次迭代：线性系统组装时间')
                R = F_int - F_ext
                from fealpy.fem import DirichletBC
                gd_uh = self.pde.dirichlet_bc
                threshold = self.pde.is_dirichlet_boundary()
                K, R = DirichletBC(self.space, gd=gd_uh,
                                threshold=threshold).apply(K, R)
                tmr.send(f'第{n}次迭代：边界条件应用时间')
                if bm.linalg.norm(R) > 1e6:
                    print("Residual too large. Exiting.")
                from fealpy.solver import spsolve
                delta_du = spsolve(K, -R, 'scipy')
                tmr.send(f'第{n}次迭代：线性系统求解时间')
                delta_u += delta_du  # 累加位移增量
                

                norm_R = bm.linalg.norm(R)
                norm_du = bm.linalg.norm(delta_du)
                print(f"  Iter {iter_count:02d}: ||R|| = {norm_R:.3e}, ||du|| = {norm_du:.3e}")

                if norm_R < tol and norm_du < tol:
                    converged = True
                    
                    stress = stress_trial
                    strain_pl = strain_pl_trial 
                    strain_e = strain_e_trial
                    print(strain_e.max())
                    strain_total = strain_total_trial
                iter_count += 1

            # 更新总位移
            u += delta_u
            print(f"  u.max = {u.max():.4e}, u.min = {u.min():.4e}")
            self.show(displacement=u, n=n)
            tmr.send(f'第{n}次后处理时间')
            next(tmr)


        return u, stress, strain_pl, strain_e
    
    def show(self, displacement, n):
        """
        Visualize the mesh and the displacement field (3D version).
        """
        save_path = "../elastoplastic_result"
        gdof = self.space.scalar_space.number_of_global_dofs()  # 节点数

        num_nodes = gdof 
        u1 = displacement[:num_nodes]
        u2 = displacement[num_nodes:2*num_nodes]
        u3 = displacement[2*num_nodes:]

        u = bm.zeros((num_nodes, 3), dtype=bm.float64)
        u[:, 0] = u1
        u[:, 1] = u2
        u[:, 2] = u3

        self.mesh.nodedata['displacement_vector'] = u
        self.mesh.to_vtk(f"{save_path}/incremental_iter_{n:03d}.vtu")

    def save_data(self):
        print("等效塑性应变:", self.strain_e)
        print("塑性应变:", self.strain_pl)
        print("应力:", self.stress)

    

       
