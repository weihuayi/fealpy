from typing import Optional, Tuple, List
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class TimoshenkoBeamMaterial(LinearElasticMaterial):
    """A class representing Timoshenko beam material properties, inheriting from LinearElasticMaterial.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the beam's geometric and material properties.

    Attributes:
        _model (object): An instance of the model, providing access to the beam's geometric properties.
        _E (float): The elastic modulus of the material.
        _mu (float): The shear modulus of the material.

    Methods:
        __init__(name: str, model): Initializes the Timoshenko beam material with the given properties.

        calculate_cross_sectional_areas() -> Tuple[TensorLike, TensorLike]:
            Calculates and returns the cross-sectional areas in the x, y, and z directions.

        calculate_moments_of_inertia() -> Tuple[TensorLike, TensorLike]:
            Calculates and returns the moments of inertia about the y, z, and polar axes.
        
        shear_factor() -> Tuple[TensorLike, TensorLike]:
            Returns the shear correction factors (FSY, FSZ) for the y and z directions.
        
        stress_matrix() -> TensorLike:
            Returns the stress matrix for the Timoshenko beam material, incorporating elastic modulus and shear effects.
        
        strain_matrix() -> TensorLike:
            Placeholder method for calculating the strain matrix (currently not implemented).
    """

    def __init__(self, name: str,
                model,
                elastic_modulus: Optional[float] = None,
                poisson_ratio: Optional[float] = None,
            ) -> None:
        super().__init__(name=name, 
                        elastic_modulus=elastic_modulus, 
                        poisson_ratio=poisson_ratio)

        self._model = model

        self._E = self.get_property('elastic_modulus')
        self._nu = self.get_property('poisson_ratio')
        self._mu = self.get_property('shear_modulus')

    def calculate_cross_sectional_areas(self) -> Tuple[TensorLike, TensorLike]:
        Ax, Ay, Az = self._model._calculate_cross_sectional_areas()
        return Ax, Ay, Az
    
    def calculate_moments_of_inertia(self) -> Tuple[TensorLike, TensorLike]:
        Iy, Iz, Ix = self._model._calculate_moments_of_inertia()
        return Iy, Iz, Ix
    
    def shear_factor(self) -> Tuple[TensorLike, TensorLike]:
        FSY, FSZ = self._model._FSY, self._model._FSZ
        return FSY, FSZ
    
    def stress_matrix(self) -> TensorLike:
        """Returns the stress matrix for Timoshenko beam material."""
        E = self._E
        mu = self._mu

        D = bm.array([[E, 0, 0],
                      [0, mu, 0],
                      [0, 0, mu]], dtype=bm.float64)
    
        return D
    
    def strain_matrix(self) -> TensorLike:  
        pass