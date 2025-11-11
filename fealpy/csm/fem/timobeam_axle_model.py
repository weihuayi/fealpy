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
from ..material import TimoshenkoBeamMaterial,  AxleMaterial
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
        
        def timo_axle_system(self):
                """Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
                self.space = LagrangeFESpace(self.mesh, self.p, ctype='C')
                self.tspace = TensorFunctionSpace(self.space, shape=(-1, 6))
                mesh = self.tspace.mesh
                n_cells = mesh.number_of_cells()

                Dofs = self.tspace.number_of_global_dofs()
                K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
                F = bm.zeros(Dofs, dtype=bm.float64)

                Timo = TimoshenkoBeamMaterial(model=self.pde,
                                        name="timobeam",
                                        elastic_modulus=self.beam_E,
                                        poisson_ratio=self.beam_nu)

                Axle = AxleMaterial(model=self.pde,
                                name="axle",
                                elastic_modulus=self.axle_E,
                                poisson_ratio=self.axle_nu)
                 
                

                timo_integrator = TimoshenkoBeamIntegrator(self.tspace, Timo, 
                                        index=bm.arange(0, n_cells-10))
                KE_beam = timo_integrator.assembly(self.tspace)
                ele_dofs_beam = timo_integrator.to_global_dof(self.tspace)

                for i, dof in enumerate(ele_dofs_beam):
                       K[dof[:, None], dof] += KE_beam[i]

                axle_integrator = AxleIntegrator(self.tspace, Axle, 
                                        index=bm.arange(n_cells-10, n_cells))
                KE_axle = axle_integrator.assembly(self.tspace)
                ele_dofs_axle = axle_integrator.to_global_dof(self.tspace)   

                for i, dof in enumerate(ele_dofs_axle):
                       K[dof[:, None], dof] += KE_axle[i]

                F[:] = self.pde.external_load()
                
                return K, F
        
        def apply_bc_penalty(self, K, F, penalty=1e20):
                """Apply Dirichlet boundary conditions using Penalty Method."""
                penalty = penalty
                fixed_dofs = bm.asarray(self.pde.dirichlet_dof_index(), dtype=int)
                
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
        
                rows, cols = bm.nonzero(K)
                values = K[rows, cols]
                K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)

                u = spsolve(K, F, solver='scipy')
                # self.logger.info(f"Solution u:\n{u}")

                return u
        
        def show(self, disp):
                """
                Visualize the mesh and the displacement field.
                """
                
                mesh = self.mesh
                
                uh = disp.reshape(-1, 6)
                u = uh[:, :3]
                mesh.nodedata['disp'] = u

                frname = f"disp.vtu"
                mesh.to_vtk(fname=frname)
                
        def calculate_strain_and_stress(self, disp, x, y, z):
                """Calculate the strain and stress.
                    ε = B * u_e
                    σ = D * ε
                    
                Parameters:
                        disp (TensorLike): Nodal displacement vector.
                        x (float): Local coordinate in the beam cross-section along the x-axis.
                        y (float): Local coordinate in the beam cross-section along the y-axis.
                        z (float): Local coordinate in the beam cross-section along the z-axis.
                        l (float): Length of the beam and axle element.
                
                Returns:
                        Tuple[TensorLike, TensorLike]: Strain and stress vectors.
                """
                mesh = self.mesh
                u = disp.rshape(-1, 6)
                
                Timo = TimoshenkoBeamMaterial(model=self.pde,
                                        name="timobeam",
                                        elastic_modulus=self.beam_E,
                                        poisson_ratio=self.beam_nu)

                Axle = AxleMaterial(model=self.pde,
                                name="axle",
                                elastic_modulus=self.axle_E,
                                poisson_ratio=self.axle_nu)
                
                l = mesh.entity_measure('cell')
                L = Timo.linear_basis(x, l)
                H = Timo.hermite_basis(x, l) 
                B = Timo.strain_matrix(x, y, z, l)
                
                e_xx = u[0]*L[0]