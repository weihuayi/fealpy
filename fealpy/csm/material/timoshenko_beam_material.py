from typing import Optional, Tuple, List
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
                model,
                elastic_modulus: Optional[float] = None,
                poisson_ratio: Optional[float] = None,
            ) -> None:
        super().__init__(name=name, 
                        elastic_modulus=elastic_modulus, 
                        poisson_ratio=poisson_ratio)

        self.model = model

        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  E (Elastic Mod.)  : {self.E}\n"
        s += f"  nu (Poisson)      : {self.nu}\n"
        s += f"  mu (Shear Mod.)   : {self.mu}\n"
        # s += f"  Shear Factors     : {self.model.FSY}, {self.model.FSZ}\n"
        # s += f"  AX, AY, AZ        : {self.model.AX.tolist()}, {self.model.AY.tolist()}, {self.model.AZ.tolist()}\n"
        # s += f"  Iy, Iz, Ix        : {self.model.Iy.tolist()}, {self.model.Iz.tolist()}, {self.model.Ix.tolist()}\n)"
        
        return s
    
    def lunzhou_material_paras(self):
        return self.model.lunzhou_E, self.model.lunzhou_mu

    def cross_sectional_areas(self) -> Tuple[TensorLike, TensorLike]:
        return self.model.AX, self.model.AY, self.model.AZ
    
    def moments_of_inertia(self) -> Tuple[TensorLike, TensorLike]:
        return self.model.Ix, self.model.Iy, self.model.Iz
    
    def shear_factor(self) -> Tuple[TensorLike, TensorLike]:
        return self.model.FSY, self.model.FSZ
    
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