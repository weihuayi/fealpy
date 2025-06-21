from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .base import MaterialInterpolation

class SIMPInterpolation(MaterialInterpolation):
    """Solid Isotropic Material with Penalization (SIMP) interpolation model."""

    def __init__(self, penalty_factor: float = 3.0):
        """Initialize SIMP interpolation."""
        if penalty_factor <= 0:
            raise ValueError("Penalty factor must be positive")
        
        super().__init__(name="SIMP")

        self.penalty_factor = penalty_factor

    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penalty_factor: float
                        ) -> TensorLike:
        """Calculate interpolated property using SIMP model."""

        if Pmin is None:
            P = rho ** penalty_factor * P0
        else:
            P = Pmin + rho ** penalty_factor * (P0 - Pmin)
        return P

    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penalty_factor: float
                                    ) -> TensorLike:
        """Calculate derivative of interpolated property using SIMP model."""

        if Pmin is None:
            dP = penalty_factor * rho ** (penalty_factor - 1) * P0
            return dP
        else:
            dP = penalty_factor * rho ** (penalty_factor - 1) * (P0 - Pmin)
            return dP

class RAMPInterpolation(MaterialInterpolation):
    """Rational Approximation of Material Properties (RAMP) interpolation model."""
    
    def __init__(self, penalty_factor: float = 3.0):
        """Initialize RAMP interpolation."""
        if penalty_factor <= 0:
            raise ValueError("Penalty factor must be positive")
        
        super().__init__(name="RAMP")

        self.penalty_factor = penalty_factor

    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penalty_factor: float
                        ) -> TensorLike:
        """Calculate interpolated property using 'RAMP' model."""

        if Pmin is None:
            P = rho * (1 + penalty_factor * (1 - rho)) ** (-1) * P0
        else:
            P = Pmin + (P0 - Pmin) * rho * (1 + penalty_factor * (1 - rho)) ** (-1)
        return P

    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penalty_factor: float
                                    ) -> TensorLike:
        """Calculate derivative of interpolated property using 'RAMP' model."""
        
        if Pmin is None:
            return P0 * (1 + penalty_factor) * (1 + penalty_factor * (1 - rho)) ** (-2)
        else:
            return (P0 - Pmin) * (1 + penalty_factor) * (1 + penalty_factor * (1 - rho)) ** (-2)