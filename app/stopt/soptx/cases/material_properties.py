from builtins import float, int, str

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

    def __repr__(self) -> str:
        """
        Return a string representation of the material properties, 
        which includes all the initialized values.
        
        Returns:
            str: A string showing the material properties.
        """
        return (f"MaterialProperties(E0={self.E0}, Emin={self.Emin}, "
                f"nu={self.nu}, penal={self.penal})")
