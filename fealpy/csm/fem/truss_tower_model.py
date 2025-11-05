from typing import Union
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

from ..model.truss import TrussPDEDataT
from ..model import CSMModelManager
from ..material import bar_meterial
from ..fem.bar_integrator import BarIntegrator


class TrussTowerModel(ComputationalModel):
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
               
    def __str__(self) -> str:
                """Returns a formatted multi-line string summarizing the configuration of the Timoshenko beam model.
                Returns:
                        str: A multi-line string containing the current model configuration,
                        displaying information such as PDE, mesh, material properties, and more.
                """
                s = f"{self.__class__.__name__}(\n"
                s += f"  pde            : {self.pde.__class__.__name__}\n"  # Assuming pde is a class object
                s += f"  mesh           : {self.mesh.__class__.__name__}\n"  # Assuming mesh is a class object
                s += f"  E           : {self.E}\n"
                s += f"  nu          : {self.nu}\n"
                s += f"  mu          : {self.E/(2*(1+self.nu)):.3e}\n"  # 自动算梁剪切模量
                s += f"  geo_dimension  : {self.GD}\n"
                s += ")"
                self.logger.info(f"\n{s}")
                return s
               
    def set_pde(self, pde: Union[TrussPDEDataT, int] = 4) -> None:
        if isinstance(pde, int):
            self.pde = CSMModelManager("truss_tower").get_example(pde)
        else:
            self.pde = pde
        # self.logger.info(f"\n{self.pde.external_load()}")
        # self.logger.info(f"\n{self.pde.dirichlet_dof()}")
                
    def set_mesh(self, mesh: Mesh) -> None:
            self.mesh = mesh
              
    def set_space_degree(self, p: int) -> None:
            self.p = p
            
    def set_material(self):
        self.material = bar_meterial(self.E, self.nu)
        
    def critical_buckling_load(self):
        """Compute critical buckling loads.
        
        Parameters:
            K: Effective length factor (dimensionless)
            E: Young's modulus (Pa)
            L: Total height (m)
            I1, I2: Area moments of inertia (m^4)
            
        Returns:
            (Fc1, Fc2) : Critical buckling loads (N)
            Fc1: Critical load for buckling about X-axis (in Y-Z plane).
            Fc2: Critical load for buckling about Y-axis (in X-Z plane)
            
        Note:
            Fc = pi^2 * E * I / (K*L)^2
        """
        K = 2.0
        node = self.mesh.entity('node')
        L = bm.max(node[:, 2]) - bm.min(node[:, 2])  # Height in z-direction
        
        I1, I2 = self.pde.structural_inertia()
        
        Fc1 = bm.pi**2 * self.E * I1 / (K*L)**2  
        Fc2 = bm.pi**2 * self.E * I2 / (K*L)**2  
        
        # self.logger.info(f"\n{L}")
        # self.logger.info(f"\n{I1}, {I2}")
        # self.logger.info(f"\n{Fc1}, {Fc2}")
        return Fc1, Fc2
            
    def truss_tower_system(self):
        pass