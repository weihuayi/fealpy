from typing import Union

from fealpy.backend import bm
from fealpy.decorator import cartesian, variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import (
        BilinearForm,  
        LinearForm,
        VectorSourceIntegrator,
        DirichletBC
        )
from fealpy.solver import spsolve

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
               
               self.set_material()
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
                # self.logger.info(f"BeamData3D : {self.pde.__str__()}")
                # self.logger.info(f"load: {self.pde.tip_load(load_case=1)}")
                # self.logger.info(f"dirichlet_dof : {self.pde.dirichlet_dof()}")
                
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
                                        poisson_ratio=self.nu,
                                        density=self.rho)
                # self.logger.info(f"Material density: {self.material.rho}") 
        
        @cartesian   
        def gravity_source(self, points):
                """Distributed gravity load function (for load case 2).
                
                Parameters:
                        points (TensorLike): Points where the load is evaluated.
                        
                Notes:
                        Gravity load intensity: q_z = -ρ * g * A (N/m)
                """
                q_z = - self.pde.Ax * self.material.rho * self.g 
                
                shape = points.shape[:-1] + (6,)
                load = bm.zeros(shape, dtype=bm.float64)
                load[..., 2] = q_z  # 仅在 z 方向上施加重力
                # self.logger.info(f"gravity: {load}")
                return load

        def assemble_load(self, load_case: int=1):
                """Assemble the complete load vector for the beam.
                
                Parameters:
                        load_case (int): The load case to assemble (1 or 2).

                Returns:
                        F(TensorLike): Global load vector with shape (gdof,).
                
                Notes:
                        Load Case 1: Concentrated tip load only.
                        Load Case 2: Distributed gravity load.
                """
                gdof = self.space.number_of_global_dofs()
                F = bm.zeros(gdof, dtype=bm.float64)
                if load_case == 1:
                        mesh = self.mesh
                        node = mesh.entity('node')
                        tip_node_idx = bm.argmax(node[:, 0])  # 最右端节点
                        load = self.pde.tip_load(load_case=1)
                
                        for i in range(6):
                                F[tip_node_idx * 6 + i] = load[i]
                elif load_case == 2:
                        lform = LinearForm(self.space)
                        # 仅在 z 方向上施加重力
                        lform.add_integrator(VectorSourceIntegrator(
                                self.gravity_source, q=self.p + 3))
                        F = lform.assembly()
                # self.logger.info(f"load: {F}")
                return F
        
        def channel_beam_system(self, load_case: int=1):
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
                F = self.assemble_load(load_case=load_case)
                # self.logger.info(f"K: {K}")
                # self.logger.info(f"load: {F}")
                return K, F

        def apply_bc(self, K, F):
                """Apply Dirichlet boundary conditions to the system."""
                
                gdof = self.space.number_of_global_dofs()
                
                threshold = bm.zeros(gdof, dtype=bool)
                fixed_dofs = self.pde.dirichlet_dof()
                threshold[fixed_dofs] = True
                
                bc = DirichletBC(
                        space=self.space,
                        gd=self.pde.dirichlet,
                        threshold=threshold
                )
                K, F = bc.apply(K, F)

                return K, F

        def solve(self, load_case: int=1):
                """Solve the linear system and return the solution.

                Returns:
                        uh: Solution vector.
                """
                gdof = self.space.number_of_global_dofs()
                K, F = self.channel_beam_system(load_case=load_case)
                
                K, F = self.apply_bc(K, F)
                uh = bm.zeros(gdof, dtype=bm.float64)
                uh = spsolve(K, F, solver='scipy')

                # self.logger.info(f"Solution : {uh}")
        
                return uh
        
        def compute_strain_and_stress(self, disp):
                """Compute axial strain and stress for beam elements."""

                uh = disp.reshape(-1, self.GD*2)

                R = self.pde.coord_transform()  # 获取变换矩阵
                
                strain, stress = self.material.compute_strain_and_stress(
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