from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.material.elastic_material import LinearElasticMaterial

from builtins import float, int, str
from typing import Optional, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

class MaterialInterpolation(ABC):
    def __init__(self, name: str):
        """
        Initialize the material interpolation model.
        """
        self.name = name

    @abstractmethod
    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, Pmin: float, 
                        penal: float) -> TensorLike:
        pass

    @abstractmethod
    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, Pmin: float, 
                                    penal: float) -> TensorLike:
        pass

@dataclass
class ElasticMaterialConfig:
    elastic_modulus: float = 1.0
    minimal_modulus: float = 1e-9
    poisson_ratio: float = 0.3
    plane_assumption: Literal["plane_stress", "plane_strain", "3d"] = "plane_stress"

class ElasticMaterialProperties(LinearElasticMaterial):
    def __init__(self, 
                 config: ElasticMaterialConfig, 
                 rho: TensorLike,
                 interpolation_model: MaterialInterpolation):
        if not isinstance(config, ElasticMaterialConfig):
            raise TypeError("'config' must be an instance of ElasticMaterialConfig")
        if rho is None or not isinstance(rho, TensorLike):
            raise TypeError("'rho' must be of type TensorLike and cannot be None")
        if interpolation_model is None or not isinstance(interpolation_model, MaterialInterpolation):
            raise TypeError("'interpolation_model' must be an instance of MaterialInterpolation and cannot be None")
        
        super().__init__(name="ElasticMaterialProperties",
                         elastic_modulus=config.elastic_modulus,
                         poisson_ratio=config.poisson_ratio,
                         hypo=config.plane_assumption)
        """
        Initialize material properties.

        This class inherits from "LinearElasticMaterial" and adds material interpolation models 
            for topology optimization.
        """
        self.config = config
        self.rho = rho
        self.interpolation_model = interpolation_model if interpolation_model else SIMPInterpolation()

    def material_model(self) -> TensorLike:
        """
        Use the interpolation model to calculate Young's modulus.
        """
        rho = self.rho
        E0 = self.config.elastic_modulus
        Emin = self.config.minimal_modulus
        penalty_factor = self.interpolation_model.penalty_factor

        E = self.interpolation_model.calculate_property(rho, 
                                                        E0, Emin, 
                                                        penalty_factor)
        return E

    def material_model_derivative(self) -> TensorLike:
        """
        Use the interpolation model to calculate the derivative of Young's modulus.
        """
        rho = self.rho
        E0 = self.config.elastic_modulus
        Emin = self.config.minimal_modulus
        penalty_factor = self.interpolation_model.penalty_factor

        dE = self.interpolation_model.calculate_property_derivative(rho,
                                                                    E0, Emin,
                                                                    penalty_factor)
        return dE

    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """
        Calculate the elastic matrix D for each element based on the density distribution.

        Returns:
            TensorLike: A tensor representing the elastic matrix D for each element.
                        Shape: (NC, 1, 3, 3) for 2D problems.
                        Shape: (NC, 1, 6, 6) for 3D problems.
        """
        if self.rho is None:
            raise ValueError("Density rho must be set for MaterialProperties.")
        
        E = self.material_model()
        base_D = super().elastic_matrix()
        D = E[:, None, None, None] * base_D
        return D

class ThermalMaterialProperties:
    def __init__(self, k0: float = 1.0, kmin: float = 1e-9, 
                 penal: int = 3, rho: Optional[TensorLike] = None, 
                 interpolation_model: MaterialInterpolation = None):
        """
        Initialize thermal material properties for topology optimization.

        Args:
            k0 (float): Thermal conductivity of the solid material.
            kmin (float): Thermal conductivity of the void or empty space.
            penal (int): Penalization factor to control thermal conductivity interpolation.
            rho (Optional[TensorLike]): Density distribution of the material (default is None).
            interpolation_model (MaterialInterpolation): Material interpolation model, 
                default is SIMP interpolation.
        """
        self.k0 = k0
        self.kmin = kmin
        self.penal = penal
        self.rho = rho
        self.interpolation_model = interpolation_model if interpolation_model else SIMPInterpolation()

    def thermal_conductivity(self) -> TensorLike:
        """
        Use the interpolation model to calculate the effective thermal conductivity.
        """
        return self.interpolation_model.calculate_property(self.rho, 
                                                        self.k0, self.kmin, self.penal)

    def thermal_conductivity_derivative(self) -> TensorLike:
        """
        Use the interpolation model to calculate the derivative of the thermal conductivity.
        """
        return self.interpolation_model.calculate_property_derivative(self.rho, 
                                                                    self.k0, self.kmin, self.penal)

class SIMPInterpolation(MaterialInterpolation):
    def __init__(self, penalty_factor: float = 3.0):
        super().__init__(name="SIMP")
        self.penalty_factor = penalty_factor

    def calculate_property(self, 
                        rho: TensorLike, 
                        P0: float, Pmin: float, 
                        penalty_factor: float) -> TensorLike:
        """
        Calculate the interpolated property using the 'SIMP' model.
        """
        if Pmin is None:
            P = rho[:] ** penalty_factor * P0
            return P
        else:
            P = Pmin + rho[:] ** penalty_factor * (P0 - Pmin)
            return P

    def calculate_property_derivative(self, 
                                    rho: TensorLike, 
                                    P0: float, Pmin: float, 
                                    penalty_factor: float) -> TensorLike:
        """
        Calculate the derivative of the interpolated property using the 'SIMP' model.
        """
        if Pmin is None:
            return penalty_factor * rho[:] ** (penalty_factor - 1) * P0
        else:
            return penalty_factor * rho[:] ** (penalty_factor - 1) * (P0 - Pmin)
