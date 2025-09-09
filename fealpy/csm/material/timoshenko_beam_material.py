from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class TimoshenkoBeamMaterial(LinearElasticMaterial):
    """Material properties for 3D Timoshenko beams.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the beam's geometric and material properties.
        E (float): The elastic modulus of the material.
        mu (float): The shear modulus of the material.
    """
    
    def __init__(self, name: str, 
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
        s += f"  [Beam]  E           : {self.E}\n"
        s += f"  [Beam]  nu          : {self.nu}\n"
        s += f"  [Beam]  mu          : {self.mu}\n"
        s += ")"
        return s
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for Timoshenko beam material."""
        E = self.E
        mu = self.mu

        D = bm.array([[E, 0, 0],
                      [0, mu, 0],
                      [0, 0, mu]], dtype=bm.float64)
    
        return D
    
    def strain_matrix(self) -> TensorLike:  
        pass