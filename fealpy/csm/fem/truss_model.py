from typing import Union
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from fealpy.decorator import variantmethod

from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm, DirichletBC, LinearForm
from fealpy.solver import spsolve

from ..fem.bar_integrator import BarIntegrator
from ..material import BarMaterial
from ..model.truss import TrussPDEDataT
from ..model.model_manager import CSMModelManager


class TrussModel(ComputationalModel):
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
            self.pde = CSMModelManager('truss').get_example(pde)
        else:
            self.pde = pde
        
        self.logger.info(self.pde)
        
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
                                    model=self.pde,
                                    elastic_modulus=self.E,
                                    poisson_ratio=self.nu)

    def linear_system(self):
        """Assemble and return stiffness matrix and load vector.

        Returns:
            K: Stiffness matrix.
            F_load: Load vector.
        """
        mesh = self.space.mesh
        
        bform = BilinearForm(self.space)
        bform.add_integrator(BarIntegrator(space=self.space, 
                                        model=self.pde, 
                                        material=self.material))
        K = bform.assembly()
        F = self.pde.load()
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
        threshold[self.pde.is_dirichlet_boundary()] = True
        bc = DirichletBC(self.space, gd=self.pde.dirichlet_bc, 
                         threshold=threshold)
        K, F = bc.apply(K, F)
        return K, F

    def solve(self, K, F):
        """
        Solve the linear system and return the solution.

        Returns:
            uh: Solution vector.
        """
        uh = spsolve(K, F, solver='scipy')
        
        # self.logger.info(f"Solution : {uh.reshape(-1, 3)}")
        
        return uh
    
    def compute_strain_and_stress(self, disp):
                """Compute axial strain and stress for truss elements."""

                uh = disp.reshape(-1, self.GD)
                
                strain, stress = self.material.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                ele_indices=None)

                # self.logger.info(f"strain: {strain}")
                # self.logger.info(f"stress: {stress}")

                return strain, stress

    def calculate_von_mises_stress(self, stress):
        """Calculate von Mises stress for truss elements."""
        mstress = self.material.calculate_mises_stress(stress)
        self.logger.info(f"mstress: {mstress}")
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

                mesh.edgedata['von_mises_stress'] = mstress
                mesh.to_vtk(f"{save_path}/von_mises_stress.vtu")