from typing import Optional, Tuple, List
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial

from ..model.beam.timoshenko_beam_data_3d import TimoshenkoBeamData3D


class TimoshenkoBeamMaterial(LinearElasticMaterial):

    def __init__(self, name: str, model) -> None:
        super().__init__(name)

        self._model = model

        self._E = self.get_property('elastic_modulus')
        self._nu = self.get_property('shear_modulus')


    def cal_A(self):
        Ax, Ay, Az = self._model._cal_A()
        return Ax, Ay, Az
    
    def cal_I(self):
        Iy, Iz, Ix = self._model._cal_I()
        return Iy, Iz, Ix
    
    def shear_factor(self):
        FSY, FSZ = self._model._FSY, self._model._FSZ
        return FSY, FSZ
    


    
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