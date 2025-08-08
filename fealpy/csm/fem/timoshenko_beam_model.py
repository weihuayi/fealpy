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
        BilinearForm,
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

               self.GD = self.pde.geo_dimension()
               self.E = options['E']
               self.nu = options['nu']
        
        def __str__(self) -> str:
             pass
               
        def set_pde(self, pde: Union[BeamPDEDataT, int] = 2) -> None:
             if isinstance(pde, int):
                self.pde = CSMModelManager("timoshenko_beam").get_example(pde)
             else:
                self.pde = pde
             #self.logger.info(self.pde)
                
        def set_mesh(self, mesh: Mesh) -> None:
              self.mesh = mesh
              #self.logger.info(self.mesh)

        def timo_beam_system(self, mesh, p: int):
                """"Construct the linear system for the 3D timoshenko beam problem.

                Parameters:
                    E (float): Young's modulus in MPa.
                    nu (float): Poisson's ratio.
                """
                self.space = LagrangeFESpace(mesh, p, ctype='C')
                model = self.set_pde()
                self.tspace = TensorFunctionSpace(self.space, shape=(-1, 6))

                TBM = TimoshenkoBeamMaterial(name="timobeam",
                                      model=model, 
                                      elastic_modulus=self.E,
                                      poisson_ratio=self.nu)

                bform = BilinearForm(self.tspace)
                bform.add_integrator(TimoshenkoBeamIntegrator(self.tspace, TBM))
                K = bform.assembly()
                F = model.external_load()
                return K, F

        @variantmethod('direct')
        def solve(self):
             pass

        @solve.register('cg')
        def solve(self):
             pass

        
