from abc import ABC, abstractmethod
from fealpy.typing import TensorLike

class MaterialInterpolation(ABC):
    """Abstract base class for material interpolation models."""
    
    def __init__(self, name: str):
        """
        初始化插值模型并设置标识名.
        
        Args:
            name : 插值模型的唯一标识符
        """
        self.name = name

    @abstractmethod
    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, 
                        Pmin: float, 
                        penal: float
                        ) -> TensorLike:
        """
        计算插值后的材料属性场.
        
        Args:
            rho : Material density field
            P0 : Property value for solid material
            Pmin : Property value for void material 
            penal : Penalization factor
            
        Returns:
            TensorLike: Interpolated property field
        """
        pass

    @abstractmethod
    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, 
                                    Pmin: float, 
                                    penal: float
                                    ) -> TensorLike:
        """
        计算插值后材料属性的导数
                
        Args:
            rho : Material density field
            P0 : Property value for solid material
            Pmin : Property value for void material
            penal : Penalization factor
            
        Returns:
            TensorLike: Derivative of interpolated property field
        """
        pass