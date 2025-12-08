from typing import Union
from fealpy.sparse import CSRTensor
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm, DirichletBC, LinearForm
from fealpy.solver import spsolve

from ..fem.bar_integrator import BarIntegrator
from ..material import BarMaterial
from ..model.truss import TrussPDEDataT
from ..model.model_manager import CSMModelManager
from ..utils import CoordTransform


class BarModel(ComputationalModel):
    """
    3D linear FEM model for truss structures with blocked DOF layout.

    Parameters:
        options (dict): Model and solver options.

    Attributes:
        options (dict): Model options.
        pde: PDE data manager.
        mesh (Mesh): The mesh object.
        space: The finite element space.
        material: Material properties.
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        A (float): Cross-sectional area.
        GD (int): Geometric dimension.

    Methods:
        set_pde(pde): Set PDE parameters and update model.
        set_mesh(mesh): Set the mesh.
        set_space(p): Set the finite element space.
        set_material(): Set material properties.
        linear_system(): Assemble and return stiffness matrix and load vector.
        solve(): Solve the linear system and return the solution.
        post_process(uh): Compute displacement, strain, and stress.
        print_results(uh): Print displacement, strain, and stress.
        run(): Solve and show results.
        show(uh/strain/stress): Export results to VTK.
    """
    def __init__(self, options):
        self.options = options
        super().__init__(
                pbar_log=options['pbar_log'],
                log_level=options['log_level'])
        self.set_pde(options['pde'])
        mesh = self.pde.init_mesh()
        self.set_mesh(mesh)
        self.set_space_degree(options['space_degree'])
        
        self.GD = self.pde.geo_dimension()

        self.E = options['E']
        self.nu = options['nu']
        
        self.set_space()
        self.set_material()
    
    def set_pde(self, pde: Union[TrussPDEDataT, int] = 3) -> None:
        """Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            if pde not in [3, 4]:
                raise ValueError(f"Invalid PDE ID: {pde}. Must be 3, 4.")
            self.pde = CSMModelManager('truss').get_example(pde)
        else:
            self.pde = pde
        
        # self.logger.info(self.pde.is_dirichlet_boundary())
        # self.logger.info(self.pde.load())
        
    def set_mesh(self, mesh: Mesh) -> None:
        """Set the mesh.

        Parameters:
            mesh (Mesh): The mesh object.
        """
        self.mesh = mesh
        
    def set_space_degree(self, p: int) -> None:
        self.p = p
        
    def set_space(self):
        """Initialize the finite element space."""
        mesh = self.mesh
        p = self.p
        
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD))

    def set_material(self) -> None:
        """Set material properties.

        Parameters:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
        """
        self.material = BarMaterial(name='BarMaterial',
                                    model=None,
                                    elastic_modulus=self.E,
                                    poisson_ratio=self.nu)

    def linear_system(self):
        """Assemble and return stiffness matrix and load vector.

        Returns:
            K: Stiffness matrix.
            F_load: Load vector.
        """
        bform = BilinearForm(self.space)
        integrator = BarIntegrator(space=self.space, 
                                        model=self.pde, 
                                        material=self.material)
        bform.add_integrator(integrator)
        K = bform.assembly()
        F = self.pde.load()
        # self.logger.info(f"strain: {K}")
        # self.logger.info(f"stress: {F}")
        return K, F
    
    def apply_bc(self, K, F):
        """Apply boundary conditions using DirichletBC.

        Parameters:
            K: Stiffness matrix.
            F: Load vector.

        Returns:
            K_bc: Modified stiffness matrix.
            F_bc: Modified load vector.
        """
        gdof = self.space.number_of_global_dofs()
        threshold = bm.zeros(gdof, dtype=bool)
         
        is_bd_dof = self.pde.is_dirichlet_boundary()
        threshold[is_bd_dof] = True
       
        gd_value = self.pde.dirichlet_bc()
        bc = DirichletBC(self.space, gd=gd_value, 
                         threshold=threshold)
        F = F.flatten()
        K, F = bc.apply(K, F)
        
        return K, F
    
    def apply_bc_penalty(self, K, F, penalty=1e12):
        """Apply Dirichlet boundary conditions using Penalty Method."""
        
        is_bd_dof = self.pde.is_dirichlet_boundary()
        fixed_dofs = bm.where(is_bd_dof)[0]
        
        F = F.flatten()
        F[fixed_dofs] *= penalty
        
        K = K.toarray()
        for dof in fixed_dofs:
                K[dof, dof] *= penalty
                
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        crow = bm.zeros(K.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K = CSRTensor(crow, cols, values, spshape=K.shape)
        
        return K, F

    def solve(self, K, F):
        """
        Solve the linear system and return the solution.

        Returns:
            uh: Solution vector.
        """
        uh = spsolve(K, F, solver='scipy')
        
        # self.logger.info(f"Solution : {uh}")
        
        return uh
    
    def compute_strain_and_stress(self, disp):
                """Compute axial strain and stress for truss elements."""

                uh = disp.reshape(-1, self.GD)
                coord_trans = CoordTransform(method='bar3d')
                R = coord_trans.coord_transform_bar3d(self.mesh)
                
                strain, stress = self.material.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                coord_transform=R,
                                ele_indices=None)

                #self.logger.info(f"strain: {strain}")
                #self.logger.info(f"stress: {stress}")

                return strain, stress

    def calculate_von_mises_stress(self, stress):
        """Calculate von Mises stress for truss elements."""
        mstress = self.material.calculate_mises_stress(stress)
        # self.logger.info(f"mstress: {mstress}")
        return mstress

    def show(self, uh, strain, stress, mstress):
                """Visualize displacement field, strain field, and stress field by saving to VTU files."""
                
                mesh = self.space.mesh
                save_path = "../truss_result"
                
                disp = uh.reshape(-1, self.GD)
        
                import os
                os.makedirs(save_path, exist_ok=True)
                
                mesh.nodedata['displacement'] = disp
                mesh.to_vtk(f"{save_path}/disp.vtu")
                
                mesh.edgedata['strain'] = strain
                mesh.to_vtk(f"{save_path}/strain.vtu")

                mesh.edgedata['stress'] = stress
                mesh.to_vtk(f"{save_path}/stress.vtu")

                mesh.edgedata['mises_stress'] = mstress
                mesh.to_vtk(f"{save_path}/mises_stress.vtu")