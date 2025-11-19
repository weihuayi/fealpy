from typing import Union
from fealpy.sparse import COOTensor

from fealpy.backend import bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import BilinearForm
from fealpy.solver import spsolve, cg

from ..model.beam import BeamPDEDataT
from ..model import CSMModelManager
from ..material import TimoshenkoBeamMaterial,  BarMaterial
from ..fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
from ..fem.axle_integrator import AxleIntegrator


class TimobeamAxleModel(ComputationalModel):
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
               self.beam_E = options['beam_E']
               self.beam_nu = options['beam_nu']
               self.axle_E = options['axle_E']
               self.axle_nu = options['axle_nu']
               
               self.Timo, self.Axle = self.set_material()
               self.set_space()
        
        def __str__(self) -> str:
                """Returns a formatted multi-line string summarizing the configuration of the Timoshenko beam model.
                Returns:
                        str: A multi-line string containing the current model configuration,
                        displaying information such as PDE, mesh, material properties, and more.
                """
                s = f"{self.__class__.__name__}(\n"
                s += "  --- Timoshenko Beam and Axle Model ---\n"
                s += f"  pde            : {self.pde.__class__.__name__}\n"  # Assuming pde is a class object
                s += f"  mesh           : {self.mesh.__class__.__name__}\n"  # Assuming mesh is a class object
                s += f"  geo_dimension  : {self.GD}\n"
                s += f"  beam_E           : {self.beam_E}\n"
                s += f"  beam_nu          : {self.beam_nu}\n"
                s += f"  beam_mu          : {self.beam_E/(2*(1+self.beam_nu)):.3e}\n"  # 自动算梁剪切模量
                s += f"  axle_E           : {self.axle_E}\n"
                s += f"  axle_nu          : {self.axle_nu}\n"
                s += f"  axle_mu          : {self.axle_E/(2*(1+self.axle_nu)):.3e}\n"  # 自动算轴承剪切模量
                s += ")"
                self.logger.info(f"\n{s}")
                return s
               
        def set_pde(self, pde: Union[BeamPDEDataT, int] = 2) -> None:
             if isinstance(pde, int):
                self.pde = CSMModelManager("timobeam_axle").get_example(pde)
             else:
                self.pde = pde
                
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
                Timo = TimoshenkoBeamMaterial(name="timobeam",
                                        model=self.pde,
                                        elastic_modulus=self.beam_E,
                                        poisson_ratio=self.beam_nu)
                
                Axle = BarMaterial(name="axle",
                                model=self.pde,
                                elastic_modulus=self.axle_E,
                                poisson_ratio=self.axle_nu)
                return Timo, Axle
                
        def timo_axle_system(self):
                """Construct the linear system for the 3D timoshenko beam problem."""
                mesh = self.mesh
                NC = mesh.number_of_cells()

                Dofs = self.space.number_of_global_dofs()
                K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
                F = bm.zeros(Dofs, dtype=bm.float64)

                timo_integrator = TimoshenkoBeamIntegrator(self.space, self.pde, self.Timo, 
                                        index=bm.arange(0, NC-10))
                KE_beam = timo_integrator.assembly(self.space)
                ele_dofs_beam = timo_integrator.to_global_dof(self.space)

                for i, dof in enumerate(ele_dofs_beam):
                       K[dof[:, None], dof] += KE_beam[i]

                axle_integrator = AxleIntegrator(self.space, self.pde, self.Axle, 
                                        index=bm.arange(NC-10, NC))
                KE_axle = axle_integrator.assembly(self.space)
                ele_dofs_axle = axle_integrator.to_global_dof(self.space)

                for i, dof in enumerate(ele_dofs_axle):
                       K[dof[:, None], dof] += KE_axle[i]

                F[:] = self.pde.external_load()
                
                return K, F
        
        def apply_bc_penalty(self, K, F, penalty=1e20):
                """Apply Dirichlet boundary conditions using Penalty Method."""
                penalty = penalty
                fixed_dofs = bm.asarray(self.pde.dirichlet_dof(), dtype=int)
                
                F[fixed_dofs] *= penalty
                for dof in fixed_dofs:
                        K[dof, dof] *= penalty
                        
                rows, cols = bm.nonzero(K)
                values = K[rows, cols]
                K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
                
                return K, F

        def solve(self):
                K, F = self.timo_axle_system()
                K, F = self.apply_bc_penalty(K, F)

                u = spsolve(K, F, solver='scipy')
                # self.logger.info(f"Solution u:\n{u}")

                return u
        
        def compute_beam_strain_and_stress(self, disp):
                """Compute axial strain and stress for beam elements."""

                uh = disp.reshape(-1, 6)
                NC = self.mesh.number_of_cells()
                beam_indices = bm.arange(0, NC-10)  # 获取前面所有梁单元的索引
                R = self.pde.coord_transform(index=beam_indices)  # 获取变换矩阵

                beam_strain, beam_stress = self.Timo.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                coord_transform=R,
                                axial_position=None,
                                ele_indices=beam_indices)
                
                # self.logger.info(f"strain: {beam_strain}")
                # self.logger.info(f"stress: {beam_stress}")
                return beam_strain, beam_stress

        def compute_axle_strain_and_stress(self, disp):
                """Compute axial strain and stress for axle elements."""
                
                uh = disp.reshape(-1, 6)
                NC = self.mesh.number_of_cells()
                axle_indices = bm.arange(NC-10, NC)  # 获取最后10个单元的索引
                
                axle_strain, axle_stress = self.Axle.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                ele_indices=axle_indices)
                
                # self.logger.info(f"strain: {axle_strain}")
                # self.logger.info(f"stress: {axle_stress}")

                return axle_strain, axle_stress
         
        def compute_strain_and_stress(self, disp):
                """Compute strain and stress for both beam and axle elements."""
                
                NC = self.mesh.number_of_cells()
                beam_strain, beam_stress = self.compute_beam_strain_and_stress(disp)
                axle_strain, axle_stress = self.compute_axle_strain_and_stress(disp)

                strain = bm.zeros((NC, 3), dtype=bm.float64)
                stress = bm.zeros((NC, 3), dtype=bm.float64)
                
                beam_indices = bm.arange(0, NC-10)
                strain[beam_indices] = beam_strain
                stress[beam_indices] = beam_stress
                
                axle_indices = bm.arange(NC-10, NC)
                strain[axle_indices] = axle_strain
                stress[axle_indices] = axle_stress
                
                # self.logger.info(f"Final strain: {strain}")
                # self.logger.info(f"Final stress: {stress}")

                return strain, stress
                
        def show(self, uh, strain, stress):
                """Visualize displacement field, strain field, and stress field by saving to VTU files."""
                
                mesh = self.space.mesh
                save_path = "../timo_axle_result"
                
                disp = uh.reshape(-1, self.GD*2)
        
                import os
                os.makedirs(save_path, exist_ok=True)
                
                mesh.nodedata['displacement'] = disp
                mesh.to_vtk(f"{save_path}/disp.vtu")
                
                mesh.edgedata['strain'] = strain
                mesh.to_vtk(f"{save_path}/strain.vtu")

                mesh.edgedata['stress'] = stress
                mesh.to_vtk(f"{save_path}/stress.vtu")

