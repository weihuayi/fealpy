from abc import ABC, abstractmethod
from fealpy.typing import TensorLike

class MaterialInterpolation(ABC):
    """Abstract base class for material interpolation models."""
    
    def __init__(self, name: str):
        """
        Initialize the material interpolation model.
        
        Args:
            name (str): Name of the interpolation model
        """
        self.name = name

    @abstractmethod
    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penal: float) -> TensorLike:
        """
        Calculate the interpolated material property.
        
        Args:
            rho (TensorLike): Material density field
            P0 (float): Property value for solid material
            Pmin (float): Property value for void material
            penal (float): Penalization factor
            
        Returns:
            TensorLike: Interpolated property field
        """
        pass

    @abstractmethod
    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penal: float) -> TensorLike:
        """
        Calculate the derivative of interpolated material property.
        
        Args:
            rho (TensorLike): Material density field
            P0 (float): Property value for solid material
            Pmin (float): Property value for void material
            penal (float): Penalization factor
            
        Returns:
            TensorLike: Derivative of interpolated property field
        """
        pass