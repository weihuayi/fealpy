from typing import Optional, Tuple, List
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class TimoshenkoBeamMaterial(LinearElasticMaterial):

    def __init__(self, name: str, model: str):
        super().__init__(name)
        self.model = model
        self 
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for Timoshenko beam material."""
        E = self.get_property('elastic_modulus')
        G = self.get_property('shear_modulus')
        FSY = self.get_property('shear_correction_factor_y')
        FSZ = self.get_property('shear_correction_factor_z')

        D = bm.array([[E, 0, 0],
                      [0, G * FSY, 0],
                      [0, 0, G * FSZ]], dtype=bm.float64)
    
        return D
    
    def strain_matrix(self) -> TensorLike:
        E = self.get_property('elastic_mpduls')
        
        pass