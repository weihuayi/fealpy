from builtins import float, int, str

from fealpy.experimental.typing import TensorLike

class MaterialProperties:
    def __init__(self, E0: float, Emin: float, nu: float, penal: int):
        """
        Initialize the material properties.

        Args:
            E0 (float): The Young's modulus for solid material.
                (e.g., the modulus of elasticity in the solid phase)
            Emin (float): The Young's modulus for void or empty space.
                (a very small value representing near-zero stiffness)
            nu (float): The Poisson's ratio.
            penal (float): The penalization factor, often used in topology optimization to control material interpolation, 
                typically in the SIMP method
        """
        self.E0 = E0       
        self.Emin = Emin   
        self.nu = nu       
        self.penal = penal         

    @staticmethod
    def material_model_SIMP(rho: TensorLike, penal: float, E0: float, Emin: float) -> TensorLike:
        """
        Material model using the SIMP approach.

        This function calculates the Young's modulus based on the density distribution 
        using the SIMP method.

        Args:
            rho (TensorLike): The density distribution of the material.
            penal (float): The penalization factor for the SIMP method.
            E0 (float): The Young's modulus of the solid material.
            Emin (float, optional): The Young's modulus of the void or empty space.

        Returns:
            TensorLike: The calculated Young's modulus based on the density distribution.
        """
        if Emin is None:
            E = rho ** penal * E0
        else:
            E = Emin + rho ** penal * (E0 - Emin)
        return E

    @staticmethod
    def material_model_SIMP_derivative(rho: TensorLike, penal: float, E0: float, Emin: float) -> TensorLike:
        """
        Derivative of the material model using the SIMP approach.

        This function calculates the derivative of the Young's modulus with respect 
        to the density distribution, which is useful for sensitivity analysis in 
        topology optimization.

        Args:
            rho (TensorLike): The density distribution of the material.
            penal (float): The penalization factor for the SIMP method.
            E0 (float): The Young's modulus of the solid material.
            Emin (float): The Young's modulus of the void or empty space.

        Returns:
            TensorLike: The derivative of the Young's modulus with respect to density.
        """
        if Emin is None:
            dE = penal * rho ** (penal - 1) * E0
        else:
            dE = penal * rho ** (penal - 1) * (E0 - Emin)
        return dE
    
    def __repr__(self) -> str:
        """
        Return a string representation of the MaterialProperties object.

        This representation includes both the basic material properties
        and indicates that different material models (e.g., SIMP) and their 
        derivatives are defined and available for use.

        Returns:
            str: A string showing the material properties and the availability
                 of different material models and their derivatives.
        """
        return (f"MaterialProperties(E0={self.E0}, Emin={self.Emin}, "
                f"nu={self.nu}, penal={self.penal}, "
                f"available_models=['SIMP', 'SIMP_derivative'])")