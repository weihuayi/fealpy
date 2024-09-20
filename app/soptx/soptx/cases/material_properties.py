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
    def calculate_modulus(self, rho: TensorLike, E0: float, Emin: float, penal: float) -> TensorLike:
        """
        Calculate the effective Young's modulus.

        This is an abstract method to be implemented by subclasses. Different interpolation methods
        like SIMP and RAMP will provide specific implementations.

        Args:
            rho (TensorLike): Density distribution of the material.
            E0 (float): Young's modulus of the solid material.
            Emin (float): Young's modulus of the void or empty space.
            penal (float): Penalization factor for the interpolation method.

        Returns:
            TensorLike: Calculated Young's modulus based on the density distribution.
        """
        pass

    @abstractmethod
    def calculate_modulus_derivative(self, rho: TensorLike, E0: float, Emin: float, penal: float) -> TensorLike:
        """
        Calculate the derivative of the Young's modulus.

        This is an abstract method to be implemented by subclasses. This derivative is used for sensitivity analysis.

        Args:
            rho (TensorLike): Density distribution of the material.
            E0 (float): Young's modulus of the solid material.
            Emin (float): Young's modulus of the void or empty space.
            penal (float): Penalization factor for the interpolation method.

        Returns:
            TensorLike: Derivative of Young's modulus with respect to density.
        """
        pass

class MaterialProperties(LinearElasticMaterial):
    def __init__(self, E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3, 
                penal: int = 3, hypo: str = 'plane_stress', 
                rho: Optional[TensorLike] = None, interpolation_model: MaterialInterpolation = None):
        """
        Initialize material properties.

        This class inherits from LinearElasticMaterial and adds material interpolation models for topology optimization.

        Args:
            E0 (float): Young's modulus of the solid material.
            Emin (float): Young's modulus of the void or empty space.
            nu (float): Poisson's ratio.
            penal (int): Penalization factor to control material interpolation.
            hypo (str): Material model hypothesis, either 'plane_stress' or '3D'.
            rho (Optional[TensorLike]): Density distribution of the material (default is None).
            interpolation_model (MaterialInterpolation): Material interpolation model, default is SIMP interpolation.
        """
        if hypo not in ["plane_stress", "3D"]:
            raise ValueError("hypo should be either 'plane_stress' or '3D'")
        
        super().__init__(name="MaterialProperties", elastic_modulus=E0, poisson_ratio=nu, hypo=hypo)

        self.E0 = E0
        self.Emin = Emin   
        self.nu = nu
        self.penal = penal
        self.hypo = hypo   
        self.rho = rho
        self.interpolation_model = interpolation_model if interpolation_model else SIMPInterpolation()

    def material_model(self) -> TensorLike:
        """
        Use the interpolation model to calculate Young's modulus.

        Returns:
            TensorLike: Young's modulus calculated using the specified interpolation model.
        """
        return self.interpolation_model.calculate_modulus(
            self.rho, 
            self.get_property('elastic_modulus'), self.Emin, 
            self.penal)

    def material_model_derivative(self) -> TensorLike:
        """
        Use the interpolation model to calculate the derivative of Young's modulus.

        Returns:
            TensorLike: Derivative of Young's modulus calculated using the specified interpolation model.
        """
        return self.interpolation_model.calculate_modulus_derivative(
            self.rho, 
            self.get_property('elastic_modulus'), self.Emin, 
            self.penal)

    def elastic_matrix(self, bcs: Optional[TensorLike] = None) -> TensorLike:
        """
        Calculate the elastic matrix D for each element based on the density distribution.

        This method utilizes the elastic matrix defined in the parent class and scales it 
        by the Young's modulus calculated using the SIMP model.

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
    
    def update_density(self, new_rho: TensorLike):
        """
        Update the density distribution for the material properties.

        Args:
            new_rho (TensorLike): The new density distribution.
        """
        if new_rho.shape != self.rho.shape:
            raise ValueError("The shape of new_rho must match the current rho shape.")
        self.rho = new_rho
    
    def __repr__(self) -> str:
        """
        Return a string representation of the MaterialProperties object.

        This representation includes basic material properties, 
        the method used for calculating the elastic matrix,
        and the currently used material interpolation model.

        Returns:
            str: A string showing the material properties.
        """
        elastic_modulus = self.get_property('elastic_modulus')
        poisson_ratio = self.get_property('poisson_ratio')
        rho_info = f"rho_shape={self.rho.shape}, rho_mean={bm.mean(self.rho):.4f}" if self.rho is not None else "rho=None"
        interpolation_model_name = self.interpolation_model.__class__.__name__

        return (f"MaterialProperties(elastic_modulus={elastic_modulus}, "
                f"Emin={self.Emin}, poisson_ratio={poisson_ratio}, "
                f"penal={self.penal}, hypo={self.hypo}, "
                f"{rho_info}, "
                f"elastic_matrix='scaled using {interpolation_model_name}', "
                f"interpolation_model={interpolation_model_name})")
    
class SIMPInterpolation(MaterialInterpolation):
    def __init__(self):
        super().__init__(name="SIMP")
        
    def calculate_modulus(self, rho: TensorLike, E0: float, Emin: float, penal: float) -> TensorLike:
        """
        Calculate the effective Young's modulus using the SIMP approach.

        This function calculates the Young's modulus based on the density distribution 
        using the SIMP method.

        Returns:
            TensorLike: The calculated Young's modulus based on the density distribution.
                        Shape: (NC, ).
        """
        if Emin is None:
            return rho[:] ** penal * E0
        else:
            return Emin + rho[:] ** penal * (E0 - Emin)

    def calculate_modulus_derivative(self, rho: TensorLike, E0: float, Emin: float, penal: float) -> TensorLike:
        """
        Derivative of the effective Young's modulus using the SIMP approach.

        This function calculates the derivative of the Young's modulus with respect 
        to the density distribution, which is useful for sensitivity analysis in 
        topology optimization.

        Returns:
            TensorLike: The derivative of the Young's modulus with respect to density.
                        Shape: (NC, ).
        """
        if Emin is None:
            return penal * rho[:] ** (penal - 1) * E0
        else:
            return penal * rho[:] ** (penal - 1) * (E0 - Emin)

