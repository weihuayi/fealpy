from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.material.elastic_material import LinearElasticMaterial

from builtins import float, int, str
from typing import Optional
from abc import ABC, abstractmethod

class MaterialInterpolation(ABC):
    def __init__(self, name: str):
        """
        Initialize the material interpolation model.

        Args:
            name (str): Name of the interpolation model.
        """
        self.name = name

    @abstractmethod
    def calculate_property(self, rho: TensorLike, 
                        P0: float, Pmin: float, penal: float) -> TensorLike:
        pass

    @abstractmethod
    def calculate_property_derivative(self, rho: TensorLike, 
                                    P0: float, Pmin: float, penal: float) -> TensorLike:
        pass

class ElasticMaterialProperties(LinearElasticMaterial):
    def __init__(self, E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3, 
                penal: int = 3, hypo: str = 'plane_stress', 
                rho: Optional[TensorLike] = None, 
                interpolation_model: MaterialInterpolation = None):
        """
        Initialize material properties.

        This class inherits from LinearElasticMaterial and adds material interpolation models 
            for topology optimization.

        Args:
            E0 (float): Young's modulus of the solid material.
            Emin (float): Young's modulus of the void or empty space.
            nu (float): Poisson's ratio.
            penal (int): Penalization factor to control material interpolation.
            hypo (str): Material model hypothesis, either 'plane_stress' or '3D'.
            rho (Optional[TensorLike]): Density distribution of the material (default is None).
            interpolation_model (MaterialInterpolation): Material interpolation model, 
                default is SIMP interpolation.
        """
        if hypo not in ["plane_stress", "3D"]:
            raise ValueError("hypo should be either 'plane_stress' or '3D'")
    
        super().__init__(name="ElasticMaterialProperties", 
                        elastic_modulus=E0, poisson_ratio=nu, hypo=hypo)
        self.E0 = E0
        self.Emin = Emin
        self.penal = penal
        self.rho = rho
        self.interpolation_model = interpolation_model if interpolation_model else SIMPInterpolation()

    def material_model(self) -> TensorLike:
        """
        Use the interpolation model to calculate Young's modulus.
        """
        return self.interpolation_model.calculate_property(self.rho, 
                                                        self.E0, self.Emin, 
                                                        self.penal)

    def material_model_derivative(self) -> TensorLike:
        """
        Use the interpolation model to calculate the derivative of Young's modulus.
        """
        return self.interpolation_model.calculate_property_derivative(self.rho, 
                                                                    self.E0, self.Emin, 
                                                                    self.penal)

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
    def __init__(self):
        super().__init__(name="SIMP")

    def calculate_property(self, rho: TensorLike, 
                        P0: float, Pmin: float, 
                        penal: float) -> TensorLike:
        """
        Calculate the interpolated property using the SIMP model.

        Args:
            rho (TensorLike): Density distribution of the material.
            P0 (float): Property value of the solid material (e.g., Young's modulus or conductivity).
            Pmin (float): Property value of the void or empty space.
            penal (float): Penalization factor for the interpolation.

        Returns:
            TensorLike: Interpolated property value based on the density distribution.
        """
        if Pmin is None:
            return rho[:] ** penal * P0
        else:
            return Pmin + rho[:] ** penal * (P0 - Pmin)

    def calculate_property_derivative(self, rho: TensorLike, 
                                    P0: float, Pmin: float, 
                                    penal: float) -> TensorLike:
        """
        Calculate the derivative of the interpolated property using the SIMP model.

        Args:
            rho (TensorLike): Density distribution of the material.
            P0 (float): Property value of the solid material.
            Pmin (float): Property value of the void or empty space.
            penal (float): Penalization factor for the interpolation.

        Returns:
            TensorLike: Derivative of the interpolated property with respect to density.
        """
        if Pmin is None:
            return penal * rho[:] ** (penal - 1) * P0
        else:
            return penal * rho[:] ** (penal - 1) * (P0 - Pmin)
