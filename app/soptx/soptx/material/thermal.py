from typing import Optional
from fealpy.typing import TensorLike
from .base import MaterialInterpolation

class ThermalMaterialProperties:
    """Class for thermal material properties with interpolation capabilities."""

    def __init__(self, 
                 k0: float = 1.0, 
                 kmin: float = 1e-9, 
                 penal: int = 3, 
                 rho: Optional[TensorLike] = None,
                 interpolation_model: Optional[MaterialInterpolation] = None):
        """
        Initialize thermal material properties.
        
        Args:
            k0 (float): Thermal conductivity of solid material
            kmin (float): Thermal conductivity of void
            penal (int): Penalization factor
            rho (Optional[TensorLike]): Density field
            interpolation_model (Optional[MaterialInterpolation]): Material interpolation model
        """
        self.k0 = k0
        self.kmin = kmin
        self.penal = penal
        self.rho = rho
        self.interpolation_model = interpolation_model

    def thermal_conductivity(self) -> TensorLike:
        """Calculate interpolated thermal conductivity."""
        return self.interpolation_model.calculate_property(
            self.rho, 
            self.k0,
            self.kmin,
            self.penal
        )

    def thermal_conductivity_derivative(self) -> TensorLike:
        """Calculate derivative of interpolated thermal conductivity."""
        return self.interpolation_model.calculate_property_derivative(
            self.rho,
            self.k0,
            self.kmin,
            self.penal
        )