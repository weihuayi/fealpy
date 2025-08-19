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
    
    def __init__(self, name: str, model,
                 elastic_modulus: Optional[float] = None,
                 poisson_ratio: Optional[float] = None,
                 shear_modulus: Optional[float] = None) -> None:
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus)

        self.model = model
    
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
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for axle material."""
        E = self.E

        D = bm.array([[E, 0, 0],
                      [0, E, 0],
                      [0, 0, E]], dtype=bm.float64)
    
        return D
    
    def strain_matrix(self) -> TensorLike:  
        pass
    