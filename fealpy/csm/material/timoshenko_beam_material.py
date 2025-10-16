from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class TimoshenkoBeamMaterial(LinearElasticMaterial):
    """Material properties for 3D Timoshenko beams.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the beam's geometric and material properties.
        E (float): The elastic modulus of the material.
        mu (float): The shear modulus of the material.
    """
    
    def __init__(self, name: str, 
                 elastic_modulus: Optional[float] = None,
                 poisson_ratio: Optional[float] = None,
                 shear_modulus: Optional[float] = None,
                 shear_factor: Optional[float] = None) -> None:
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus)
        
        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')
        self.kappa = shear_factor
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [Beam]  E           : {self.E}\n"
        s += f"  [Beam]  nu          : {self.nu}\n"
        s += f"  [Beam]  mu          : {self.mu}\n"
        s += ")"
        return s

    def strain_matrix(self, xi: float, l: float, y: float, z: float) -> TensorLike:
        """ Construct the strain-displacement matrix for a 3D Timoshenko beam element.
        This matrix relates the element nodal displacement vector to the strain vector
        at a specific local position (xi, y, z) within the beam element:
            ε = B * u_e

        Generalized strain components: ε = [ ε_x, γ_xy, γ_xz ]^T
            ε_x   : axial normal strain
            γ_xy  : shear strain in x-y plane
            γ_xz  : shear strain in x-z plane

        Parameters:
            xi(float): Local coordinate along the beam axis (ξ ∈ [0, 1]).
            l(float): Length of the beam element.
            y(float):  Local coordinate in the beam cross-section along the y-axis.
            z(float):  Local coordinate in the beam cross-section along the z-axis.

        Returns:
            B(TensorLike): Strain-displacement matrix, shape (3, 12).
        """
        # Linear shape functions
        N0 = 1.0 - xi
        N1 = xi
        
        # dN_i/dx = dN_i/dξ * dξ/dx = (±1)/l
        N0_x = -1.0 / l # d(N0)/dx
        N1_x = 1.0 / l  # d(N1)/dx

        B0 = bm.zeros((3, 6), dtype=bm.float64)
        B1 = bm.zeros((3, 6), dtype=bm.float64)
        
        B0[0, 0] = N0 
        B0[0, 4] = z * N0_x 
        B0[0, 5] = -y * N0_x
        B0[1, 1] = N0_x
        B0[1, 5] = -N0
        B0[2, 2] = N0_x
        B0[2, 4] = N0

        B1[0,0] = N1
        B1[0,4] = z * N1_x
        B1[0,5] = -y * N1_x
        B1[1,1] = N1_x
        B1[1,5] = -N1
        B1[2,2] = N1_x
        B1[2,4] = N1

        B = bm.concatenate([B0, B1], axis=1)

        return B    
    
    def stress_matrix(self) -> TensorLike:
        """Construct the stress-strain matrix D for a 3D Timoshenko beam element.
        This matrix defines the linear elastic relationship between stress and strain:
                    σ = D * ε
        where:
            σ = [ σ_x, τ_xy, τ_xz ]^T
            
        Parameters:
            kappa(float): Shear correction factor.Typical values:
                Solid rectangular section: κ ≈ 5/6,
                Circular section: κ ≈ 9/10.
                    
        Returns:
            D(TensorLike): Stress-strain matrix, shape (3, 3).
                    [ [E,     0,        0     ],
                    [0,  G/κ,        0     ],
                    [0,     0,     G/κ   ] ]
        """
        E = self.E
        mu = self.mu
        
        D = bm.array([[E, 0, 0],
                      [0, mu*self.kappa, 0],
                      [0, 0, mu*self.kappa]], dtype=bm.float64)

        return D
    
    def calculate_strain_and_stress(self,
                    displacement: TensorLike,
                    xi: float,
                    l: float,
                    y: float,
                    z: float) -> Tuple[TensorLike, TensorLike]:
        """Calculate the strain and stress.
                    ε = B * u_e
                    σ = D * ε
                    
        Parameters:
            xi (float): Local coordinate along the beam axis (ξ ∈ [0, 1]).
            displacement (TensorLike): Nodal displacement vector.
            l (float): Length of the beam element.
            y (float): Local coordinate in the beam cross-section along the y-axis.
            z (float): Local coordinate in the beam cross-section along the z-axis.

        Returns:
            Tuple[TensorLike, TensorLike]: Strain and stress vectors.
        """
        B = self.strain_matrix(xi, l, y, z)
        strain = B @ displacement
        stress = self.stress_matrix() @ strain
        return strain, stress