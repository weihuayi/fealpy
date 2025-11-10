from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class AxleMaterial(LinearElasticMaterial):
    """Material properties for 3D axles.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the axle's geometric and material properties.
        E (float): The elastic modulus of the material.
        mu (float): The shear modulus of the material.
    """
    
    def __init__(self, 
                model,
                name: str, 
                elastic_modulus: Optional[float] = None,
                poisson_ratio: Optional[float] = None,
                shear_modulus: Optional[float] = None) -> None:
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus)

        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [Axle]  E           : {self.E}\n"
        s += f"  [Axle]  nu          : {self.nu}\n"
        s += f"  [Axle]  mu          : {self.mu}\n"
        s += ")"
        return s
    
    def linear_basis(self, x: float, l: float) -> TensorLike:
        """Linear shape functions for a axle material.

        Parameters:
            x (float): Local coordinate along the axle axis.
            l (float): Length of the axle element.

        Returns:
            b (TensorLike): Linear shape functions evaluated at xi.
        """
        xi = x / l  
        t = 1.0 / l
        
        b = bm.zeros((2, 2), dtype=bm.float64)
        
        b[0, 0] = 1 - xi
        b[0, 1] = xi
        b[1, 0] = -t
        b[1, 1] = t
        return b
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for axle material."""
        E = self.E

        D = bm.array([[E, 0, 0],
                      [0, E, 0],
                      [0, 0, E]], dtype=bm.float64)
    
        return D