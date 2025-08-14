from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.csm.model.beam.timoshenko_beam_data_3d import TimoshenkoBeamData3D


class TimoshenkoBeamMaterial(LinearElasticMaterial):
    """Material properties for 3D Timoshenko beams.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the beam's geometric and material properties.
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
        
        # Beam material
        self.beam_E = self.get_property('elastic_modulus')
        self.beam_nu = self.get_property('poisson_ratio')
        self.beam_mu = self.get_property('shear_modulus')
        
        # Axle material
        self.axle_E = self.get_property('elastic_modulus')
        self.axle_nu = self.get_property('poisson_ratio')
        self.axle_mu = self.get_property('shear_modulus')
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [Beam]  E           : {self.beam_E}\n"
        s += f"  [Beam]  nu          : {self.beam_nu}\n"
        s += f"  [Beam]  mu          : {self.beam_mu}\n"
        s += f"  [Axle]  E           : {self.axle_E}\n"
        s += f"  [Axle]  mu          : {self.axle_mu}\n"
        s += ")"
        return s

    def beam_cross_section(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        return self.model.beam_Ax, self.model.beam_Ay, self.model.beam_Az

    def beam_inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        return self.model.beam_Ix, self.model.beam_Iy, self.model.beam_Iz

    def shear_factor(self) -> Tuple[TensorLike, TensorLike]:
        return self.model.FSY, self.model.FSZ
    
    def beam_stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for Timoshenko beam material."""
        beam_E = self.beam_E
        beam_mu = self.beam_mu

        beam_D = bm.array([[beam_E, 0, 0],
                      [0, beam_mu, 0],
                      [0, 0, beam_mu]], dtype=bm.float64)
    
        return beam_D
    
    def axle_stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for axle material."""
        axle_E = self.axle_E

        axle_D = bm.array([[axle_E, 0, 0],
                      [0, axle_E, 0],
                      [0, 0, axle_E]], dtype=bm.float64)
    
        return axle_D
    
    def strain_matrix(self) -> TensorLike:  
        pass