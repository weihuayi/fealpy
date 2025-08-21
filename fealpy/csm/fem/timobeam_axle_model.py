from typing import Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import (
        LagrangeFESpace, 
        TensorFunctionSpace
        )
from fealpy.fem import (
        BilinearForm,LinearForm,
        VectorSourceIntegrator,
        DirichletBC
        )
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
                self.pde = CSMModelManager("timoshenko_beam").get_example(pde)
             else:
                self.pde = pde
                
        def set_mesh(self, mesh: Mesh) -> None:
              self.mesh = mesh
              
        def set_space_degree(self, p: int) -> None:
                self.p = p

        def timo_axle_system(self):
                """"Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
                self.space = LagrangeFESpace(self.mesh, self.p, ctype='C')
                self.tspace = TensorFunctionSpace(self.space, shape=(-1, 6))

                Timo = TimoshenkoBeamMaterial(name="timobeam",
                                      model=self.pde, 
                                      elastic_modulus=self.beam_E,
                                      poisson_ratio=self.beam_nu)
                
                Axle = AxleMaterial(name="axle",
                                      model=self.pde, 
                                      elastic_modulus=self.axle_E,
                                      poisson_ratio=self.axle_nu)
                 
                 
                mesh = self.tspace.mesh
                bform_beam = BilinearForm(self.tspace)
                bform_beam.add_integrator(TimoshenkoBeamIntegrator(self.tspace, Timo, 
                                                index=bm.arange(0, mesh.number_of_cells()-10)))
                beam_K = bform_beam.assembly(format='csr')

                bform_axle = BilinearForm(self.tspace)
                bform_axle.add_integrator(AxleIntegrator(self.tspace, Axle, 
                                                index=bm.arange(mesh.number_of_cells()-10, mesh.number_of_cells())))
                axle_K = bform_axle.assembly(format='csr')

                # 直接相加
                K = beam_K + axle_K
                F = self.pde.external_load()        

                return K, F
        
        def apply_bc(self, K, F):
                """Apply boundary conditions to the linear system."""
                num_dofs = self.tspace.number_of_global_dofs()
                threshold = bm.zeros(num_dofs, dtype=bool)
                
                fixed_dofs = bm.asarray(self.pde.dirichlet_dof_index(), dtype=int)
                threshold[fixed_dofs] = True
                
                K, F = DirichletBC(space=self.tspace,
                                gd=self.pde.dirichlet,
                                threshold=threshold).apply(K, F)
                return K, F
        
        def apply_bc_penalty(self, K, F):
                """Apply Dirichlet boundary conditions using Penalty Method."""
                penalty = 1e20
                fixed_dofs = bm.asarray(self.pde.dirichlet_dof_index(), dtype=int)
        
                F[fixed_dofs] *= penalty

                crow, col, values = K._crow, K._col, K._values
                for dof in fixed_dofs:
                        row_start, row_end = crow[dof], crow[dof+1]
                        for idx in range(row_start, row_end):
                                if col[idx] == dof:   # 找到对角线位置
                                        values[idx] *= penalty
                                        break     
                
                return K, F

        @variantmethod("direct")
        def solve(self):
                K, F = self.timo_axle_system()
                # K, F = self.apply_bc(K, F)
                K, F = self.apply_bc_penalty(K, F)
                return  spsolve(K, F, solver='scipy')  