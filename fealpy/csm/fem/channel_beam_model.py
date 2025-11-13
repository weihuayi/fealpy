from typing import Union

from fealpy.backend import bm
from fealpy.decorator import cartesian, variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import BilinearForm, DirichletBC
from fealpy.solver import spsolve, cg

from ..model.beam import BeamPDEDataT
from ..model import CSMModelManager
from ..material import TimoshenkoBeamMaterial
from ..fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator


class ChannelBeamModel(ComputationalModel):
        """
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
               self.rho = options['rho']
               self.g = options['g']
               
               self.material = self.set_material()
               self.set_space()
               
        
        def __str__(self) -> str:
                """Returns a formatted multi-line string summarizing the configuration of the Timoshenko beam model.
                Returns:
                        str: A multi-line string containing the current model configuration,
                        displaying information such as PDE, mesh, material properties, and more.
                """
                s = f"{self.__class__.__name__}(\n"
                s += "  --- Channel Beam Model ---\n"
                s += f"  pde            : {self.pde.__class__.__name__}\n"  # Assuming pde is a class object
                s += f"  mesh           : {self.mesh.__class__.__name__}\n"  # Assuming mesh is a class object
                s += f"  geo_dimension  : {self.GD}\n"
                s += f"  E           : {self.E}\n"
                s += f"  nu          : {self.nu}\n"
                s += f"  mu          : {self.E/(2*(1+self.nu)):.3e}\n" 
                s += f"  rho         : {self.rho}\n"
                s += ")"
                self.logger.info(f"\n{s}")
                return s
               
        def set_pde(self, pde: Union[BeamPDEDataT, int] = 3) -> None:
                if isinstance(pde, int):
                        self.pde = CSMModelManager("channel_beam").get_example(pde)
                else:
                        self.pde = pde
                # self.logger.info(f"Solution : {self.pde.__str__()}")
                self.logger.info(f"Solution : {self.pde. tip_load(load_case=1)}")
                self.logger.info(f"Solution : {self.pde.dirichlet_dof()}")
                
        def set_mesh(self, mesh: Mesh) -> None:
                self.mesh = mesh
              
        def set_space_degree(self, p: int) -> None:
                self.p = p
                
        def set_space(self):
                """Initialize the finite element space."""
                mesh = self.mesh
                p = self.p
                
                scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
                self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD*2))
        
        def set_material(self) -> None:
                self.material = TimoshenkoBeamMaterial(name="timobeam",
                                        model=self.pde,
                                        elastic_modulus=self.E,
                                        poisson_ratio=self.nu)
                
        @cartesian
        def source(self, points, index=None):
                """Distributed load function (gravity load in load case 2).
                
                Parameters:
                points(TensorLike): Spatial points where loads are evaluated, shape (..., GD).
                index: Element index for evaluation, default is None.
                
                Notes:
                The gravity load is applied in the negative z-direction (downward).
                The load vector has 6 components: [Fx, Fy, Fz, Mx, My, Mz],  
                Only Fz is non-zero for gravity load.
                """
                q_z = -self.rho * self.g * self.pde.A  # gravity load
                
                shape = points.shape[:-1] + (6,)
                load = bm.zeros(shape, dtype=bm.float64)
                load[..., 2] = q_z  # force in z-direction
                
                return load
        
        def channel_beam_system(self):
                """Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
                bform = BilinearForm(self.space)
                bform.add_integrator(TimoshenkoBeamIntegrator(space=self.space, 
                                        model=self.pde, 
                                        material=self.material))
                K = bform.assembly()
                F = self.pde.external_load()
                
                return K, F
        
        def apply_bc(self, K, F):
                """Apply Dirichlet boundary conditions to the system."""
                
                gdof = self.space.number_of_global_dofs()
                
                threshold = bm.zeros(gdof, dtype=bool)
                fixed_dofs = self.pde.dirichlet_dof()
                threshold[fixed_dofs] = True
                
                bc = DirichletBC(
                        space=self.space,
                        gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                        threshold=threshold
                )
                K, F = bc.apply(K, F)

                return K, F

        def solve(self):
                """Solve the linear system and return the solution.

                Returns:
                        uh: Solution vector.
                """
                gdof = self.space.number_of_global_dofs()
                K, F = self.channel_beam_system()
                
                
                K, F = self.apply_bc(K, F)
                uh = bm.zeros(gdof, dtype=bm.float64)
                uh = spsolve(K, F, solver='scipy')

                # self.logger.info(f"Solution : {uh}")
        
                return uh
        
        def compute_strain_and_stress(self, disp):
                """Compute axial strain and stress for beam elements."""

                uh = disp.reshape(-1, self.GD*2)

                timo_integrator = TimoshenkoBeamIntegrator(self.space, self.pde, self.material)
                R = timo_integrator._coord_transform()  # 获取变换矩阵
                
                strain, stress = self.Timo.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                coord_transform=R,
                                axial_position=None,
                                ele_indices=None)
                
                # self.logger.info(f"strain: {strain}")
                # self.logger.info(f"stress: {stress}")
                return strain, stress
                
        def show(self, uh, strain, stress):
                """Visualize displacement field, strain field, and stress field by saving to VTU files."""
                
                mesh = self.space.mesh
                save_path = "../channel_beam_result"
                
                disp = uh.reshape(-1, self.GD*2)
        
                import os
                os.makedirs(save_path, exist_ok=True)
                
                mesh.nodedata['displacement'] = disp
                mesh.to_vtk(f"{save_path}/disp.vtu")
                
                mesh.edgedata['strain'] = strain
                mesh.to_vtk(f"{save_path}/strain.vtu")

                mesh.edgedata['stress'] = stress
                mesh.to_vtk(f"{save_path}/stress.vtu")