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
                s += f"  pde            : {self.pde.__class__.__name__}\n"  # Assuming pde is a class object
                s += f"  mesh           : {self.mesh.__class__.__name__}\n"  # Assuming mesh is a class object
                s += f"  beam_E           : {self.beam_E}\n"
                s += f"  beam_nu          : {self.beam_nu}\n"
                s += f"  beam_mu          : {self.beam_E/(2*(1+self.beam_nu)):.3e}\n"  # 自动算梁剪切模量
                s += f"  axle_E           : {self.axle_E}\n"
                s += f"  axle_nu          : {self.axle_nu}\n"
                s += f"  axle_mu          : {self.axle_E/(2*(1+self.axle_nu)):.3e}\n"  # 自动算轴承剪切模量
                s += f"  geo_dimension  : {self.GD}\n"
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
                """Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
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
        
        def compute_beam_strain_and_stress(self,
                                disp,
                                cross_section_coords=(0.0, 0.0),
                                evaluation_point=None):
                uh = disp.reshape(-1, 6)
                
                mesh = self.mesh
                cells = mesh.entity('cell')  # shape: (NC, 2)
                NC = mesh.number_of_cells()
                lengths = mesh.entity_measure('cell')
                
                beam_elements = NC - 10  # 梁单元数量
                
                if evaluation_point is None:
                        mid_x = lengths / 2.0
                else:
                        mid_x = evaluation_point

                y, z = cross_section_coords  # 截面局部坐标
                
                
                timo_integrator = TimoshenkoBeamIntegrator(self.space, self.pde, self.Timo, 
                                        index=bm.arange(0, beam_elements))
                R = timo_integrator._coord_transform()  # 获取变换矩阵
                
                beam_strain = bm.zeros((beam_elements, 3))
                beam_stress = bm.zeros((beam_elements, 3))
                
                for i in range(beam_elements):
                        node_indices = cells[i]  # [node0_idx, node1_idx]
                        element_disp = bm.concatenate([
                                uh[node_indices[0]], 
                                uh[node_indices[1]]
                                ])  # shape: (12,)
                        
                        # 提取局部位移和转角
                        local_disp = R[i] @ element_disp  # shape: (12,)
                        u0 = local_disp[0:3]
                        a0 = local_disp[3:6]
                        u1 = local_disp[6:9]
                        a1 = local_disp[9:12]
                        
                        L = self.Timo.linear_basis(mid_x[i], lengths[i])
                        H = self.Timo.hermite_basis(mid_x[i], lengths[i], index=i)

                        # 轴向应变: εxx = ∂u/∂x - y*∂θz/∂x + z*∂θy/∂x
                        e_xx = (u0[0]*L[1, 0] + u1[0]*L[1, 1] -
                                y*(u0[1]*H[1, 0] + a0[2]*H[1, 1] + u1[1]*H[1, 2] + a1[2]*H[1, 3]) +
                                z*(-u0[2]*H[1, 4] + a0[1]*H[1, 5] - u1[2]*H[1, 6] + a1[1]*H[1, 7]))
                        
                        # xy平面剪切应变: γxy = ∂v/∂x - θz
                        dv_dx = u0[1]*H[1, 0] + a0[2]*H[1, 1] + u1[1]*H[1, 2] + a1[2]*H[1, 3]  # ∂v/∂x
                        theta_z = u0[1]*H[0, 0] + a0[2]*H[0, 1] + u1[1]*H[0, 2] + a1[2]*H[0, 3]  # θz插值
                        e_xy = dv_dx - theta_z
                        
                        # xz平面剪切应变: γxz = ∂w/∂x + θy  
                        dw_dx = u0[2]*H[1, 4] + a0[1]*H[1, 5] + u1[2]*H[1, 6] + a1[1]*H[1, 7]  # ∂w/∂x
                        theta_y = -u0[2]*H[0, 4] - a0[1]*H[0, 5] - u1[2]*H[0, 6] - a1[1]*H[0, 7]  # θy插值
                        e_xz = dw_dx + theta_y
                        
                        beam_strain[i, 0] = e_xx
                        beam_strain[i, 1] = e_xy
                        beam_strain[i, 2] = e_xz
                        
                        # 应力计算（方法一）
                        # kappa = 10/9
                        # beam_stress[i, 0] = self.Timo.E * beam_strain[i, 0]
                        # beam_stress[i, 1] = kappa*self.Timo.mu * beam_strain[i, 1]
                        # beam_stress[i, 2] = kappa*self.Timo.mu * beam_strain[i, 2]

                # 应力计算（方法二）
                beam_stress = beam_strain @ self.Timo.stress_matrix()

                # self.logger.info(f"strain:\n{beam_strain.shape}")
                # self.logger.info(f"strain:\n{beam_stress.shape}")
                # self.logger.info(f"strain:\n{beam_strain}")
                # self.logger.info(f"strain:\n{beam_stress}")

                return beam_strain, beam_stress

        def compute_axle_strain_and_stress(self, disp):
                """Compute axial forces for axle elements."""
                
                uh = disp.reshape(-1, 6)
                NC = self.mesh.number_of_cells()
                axle_indices = bm.arange(NC-10, NC)  # 获取最后10个单元的索引
                
                axle_strain, axle_stress = self.Axle.compute_strain_and_stress(
                                self.mesh,
                                uh,
                                ele_indices=axle_indices)
                
                self.logger.info(f"Final strain shape: {axle_strain}")
                self.logger.info(f"Final stress shape: {axle_stress}")

                return axle_strain, axle_stress
         

        def show(self, disp, strain, stress):
                """Visualize mesh, displacement field, strain field, and stress field by saving to VTU files."""
                
                mesh = self.mesh
                
                uh = disp.reshape(-1, 6)
                u = uh[:, :3]
                mesh.nodedata['disp'] = u

                frname = f"disp.vtu"
                mesh.to_vtk(fname=frname)

                mesh.edgedata['strain'] = strain
                frname = f"strain.vtu"
                mesh.to_vtk(fname=frname)

                mesh.edgedata['stress'] = stress
                frname = f"stress.vtu"
                mesh.to_vtk(fname=frname)
