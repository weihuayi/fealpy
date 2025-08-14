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
        ScalarSourceIntegrator,
        DirichletBC
        )
from fealpy.solver import spsolve, cg

from ..model.beam import BeamPDEDataT
from ..model import CSMModelManager
from ..material import TimoshenkoBeamMaterial
from ..fem import TimoshenkoBeamIntegrator


class TimoshenkoBeamModel(ComputationalModel):
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

        def timo_beam_system(self):
                """"Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
                self.space = LagrangeFESpace(self.mesh, self.p, ctype='C')
                self.tspace = TensorFunctionSpace(self.space, shape=(-1, 6))

                TBM = TimoshenkoBeamMaterial(name="timobeam",
                                      model=self.pde, 
                                      elastic_modulus=self.beam_E,
                                      poisson_ratio=self.beam_nu)

                bform = BilinearForm(self.tspace)
                bform.add_integrator(TimoshenkoBeamIntegrator(self.tspace, TBM))
                
                K = bform.assembly()
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
        
        
        @variantmethod("direct")
        def solve(self):
                K, F = self.timo_beam_system()
                K, F = self.apply_bc(K, F)
                return  spsolve(K, F, solver='scipy') 