from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial

from ..model.beam.timobeam_axle_data_3d import TimobeamAxleData3D


class TimoshenkoBeamMaterial(LinearElasticMaterial):
    """Material properties for 3D Timoshenko beams.

    Parameters:
        name (str): The name of the material.
        model (object): The model containing the beam's geometric and material properties.
        E (float): The elastic modulus of the material.
        mu (float): The shear modulus of the material.
    """
    
    def __init__(self, 
                model,
                name: str, 
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

        model = TimobeamAxleData3D()

        self.A = model.beam_Ax  # 截面面积
        self.Iy = model.beam_Iy  # 绕 y 轴惯性矩
        self.Iz = model.beam_Iz  # 绕 z 轴惯性矩
        self.J = model.beam_Ix  # 极惯性矩
        
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [Beam]  E           : {self.E}\n"
        s += f"  [Beam]  nu          : {self.nu}\n"
        s += f"  [Beam]  mu          : {self.mu}\n"
        s += ")"
        return s

    def linear_basis(self, x: float, l: float) -> TensorLike:
        """Linear shape functions for a 3D Timoshenko beam element.

        Parameters:
            x (float): Local coordinate along the beam axis.
            l (float): Length of the beam element.

        Returns:
            b (TensorLike): Linear shape functions evaluated at xi.
        """
        xi = x / l  
        t = 1.0 / l
        
        b = bm.zeros((2, 2), dtype=bm.float64)
        
        b[0, 0] = 1 - xi
        b[0, 1] = -t
        b[1, 0] = xi
        b[1, 1] = t
        return b
    
    def hermite_basis(self, x: float, l: float) -> TensorLike:
        """Hermite shape functions for a 3D Timoshenko beam element.

        Parameters:
            x (float): Local coordinate along the beam axis.
            l (float): Length of the beam element.

        Returns:
            b (TensorLike): Hermite shape functions evaluated at xi.
        """
        t0 = 1.0 / l
        t1 = x / l
        t2 = t1 **2
        t3 = t1 **3
        
        h = bm.zeros((3, 4), dtype=bm.float64)
        
        h[0, 0] = 1- 3*t2 + 2*t3
        h[0, 1] = x - 2*x*t1 + x*t2
        h[0, 2] = 3*t2 - 2*t3
        h[0, 3] = -x*t1 +x*t2
        
        # 一阶导数
        h[1, 0] = 6*t0 * (t2 - t1)
        h[1, 1] = 1 - 4*t1 + 3*t2
        h[1, 2] = 6*t0 * (t1 - t2)
        h[1, 3] = 3*t2 - 2*t1

        # 二阶导数
        h[2, 0] = t0**2 * (12*t1 - 6)
        h[2, 1] = t0 * (6*t1 - 4)
        h[2, 2] = t0**2 * (6 - 12*t1)
        h[2, 3] = t0 * (6*t1 - 2)
        return h

    def shape_function(self, x: float, l: float, plane='xy') -> float:
        """Construct the Hermite shape function blocks for bending and rotation.

        Parameters:
            l (float): Length of the beam element.
            plane (str, optional): Plane of interest ('xy', 'zx').

        Returns:
            N  : (4,4) shape function matrix
            Nt : (4,4) derivative matrix
        """
        if plane == 'xy':
            Lambda = self.E * self.Iz / (self.kappa * self.mu * self.A)
        elif plane == 'zx':
            Lambda = self.E * self.Iy / (self.kappa * self.mu * self.A)
        else:
            raise ValueError("plane must be 'yz' or 'xz'")
        
        phi = 12 * Lambda / l**2
        psi = 1 / (1 + phi)
        
        xi = x / l
        N0 = psi * (1 - 3*xi**2 + 2*xi**3 + phi*(1 - xi))
        N1 = l*psi * (xi - 2*xi**2 +xi**3 + phi*(xi-xi**2)/2)
        N2 = psi * (3*xi**2 -2*xi**3 + phi*xi)
        N3 = l*psi * (-xi**2 + xi**3 + phi*(-xi+xi**2)/2)
        
        Nt0 =  6*psi / l*(-xi + xi**2)
        Nt1 = psi * (1 - 4*xi + 3*xi**2 + phi*(1 - xi))
        Nt2 = -6*psi / l*(-xi + xi**2)
        Nt3 = psi * (-2*xi + 3*xi**2 + phi*xi)
        
        N = bm.concatenate([N0, N1, N2, N3], axis=0)
        Nt = bm.concatenate([Nt0, Nt1, Nt2, Nt3], axis=0)

        return N, Nt

    def strain_matrix(self, x: float, l: float, plane='xy') -> TensorLike:
        """ Compute the strain-displacement matrix [B] for 3D Timoshenko beam element.

        Parameters:
            x (float): Local coordinate along the beam axis.
            l (float): Length of the beam element.
            plane (str, optional): Plane of interest ('xy' or 'zx').

        Returns:
            B (TensorLike): Strain-displacement matrix, shape (3, 12).
        """
        L = self.linear_basis(x, l)
        N, Nt = self.shape_function(x, l, plane=plane)
        
        B0 = bm.zeros((3, 6), dtype=bm.float64)
        B1 = bm.zeros((3, 6), dtype=bm.float64)

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
        kappa = self.kappa
        
        D = bm.array([[E, 0, 0],
                      [0, mu*kappa, 0],
                      [0, 0, mu*kappa]], dtype=bm.float64)

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