from fealpy.experimental.backend import backend_manager as bm

from builtins import float, int, str
from typing import Optional
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.material.elastic_material import LinearElasticMaterial

class MaterialProperties(LinearElasticMaterial):
    def __init__(self, E0: float, Emin: float, nu: float, penal: int, hypo: str, 
                rho: Optional[TensorLike] = None):
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
            hypo (str): The hypothesis for the material model, either 'plane_stress' or '3D'.
            rho (Optional[TensorLike]): The density distribution of the material (default is None).


        """
        if hypo not in ["plane_stress", "3D"]:
            raise ValueError("hypo should be either 'plane_stress' or '3D'")
        
        super().__init__(name="MaterialProperties", elastic_modulus=E0, poisson_ratio=nu, hypo=hypo)

        self.Emin = Emin   
        self.penal = penal
        self.hypo = hypo   
        self.rho = rho       

    def material_model_SIMP(self) -> TensorLike:
        """
        Calculate the effective Young's modulus using the SIMP approach.

        This function calculates the Young's modulus based on the density distribution 
        using the SIMP method.

        Returns:
            TensorLike: The calculated Young's modulus based on the density distribution.
                        Shape: (NC, ).
        """
        Emin = self.Emin
        penal = self.penal
        rho = self.rho
        E0 = self.get_property('elastic_modulus')
        if Emin is None:
            E = rho ** penal * E0
        else:
            E = Emin + rho ** penal * (E0 - Emin)
        return E

    def material_model_SIMP_derivative(self) -> TensorLike:
        """
        Derivative of the effective Young's modulus using the SIMP approach.

        This function calculates the derivative of the Young's modulus with respect 
        to the density distribution, which is useful for sensitivity analysis in 
        topology optimization.

        Returns:
            TensorLike: The derivative of the Young's modulus with respect to density.
                        Shape: (NC, ).
        """
        Emin = self.Emin
        penal = self.penal
        rho = self.rho
        E0 = self.get_property('elastic_modulus')
        if Emin is None:
            dE = penal * rho ** (penal - 1) * E0
        else:
            dE = penal * rho ** (penal - 1) * (E0 - Emin)
        return dE
    
    def elastic_matrix(self) -> TensorLike:
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
        
        E = self.material_model_SIMP()

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

        This representation includes both the basic material properties, 
        the approach to calculating the elastic matrix using the SIMP method,
        and indicates that different material models (e.g., SIMP) and their 
        derivatives are defined and available for use.

        Returns:
            str: A string showing the material properties, the approach to 
                elastic matrix calculation, and the availability of different
                material models and their derivatives.
        """
        elastic_modulus = self.get_property('elastic_modulus')
        poisson_ratio = self.get_property('poisson_ratio')
        rho_info = f"rho_shape={self.rho.shape}, rho_mean={bm.mean(self.rho):.4f}" if self.rho is not None else "rho=None"

        return (f"MaterialProperties(elastic_modulus={elastic_modulus}, "
                f"Emin={self.Emin}, poisson_ratio={poisson_ratio}, "
                f"penal={self.penal}, hypo={self.hypo}, "
                f"{rho_info}, "
                f"elastic_matrix='scaled using SIMP', "
                f"available_models=['SIMP', 'SIMP_derivative'])")
