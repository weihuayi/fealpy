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
        E (float): Young's modulus (elastic_modulus).
        nu (float): Poisson's ratio.
        lam (float): Lamé's first parameter (λ).
        mu (float): Shear modulus (μ).
        rho (float): Material density.
    """
    
    def __init__(self, 
                name: str, 
                model,
                elastic_modulus: Optional[float] = None,
                poisson_ratio: Optional[float] = None,
                shear_modulus: Optional[float] = None,
                density: Optional[float] = None) -> None:
        
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus,
                        density=density)

        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')
        self.rho = self.get_property('density')
        
        self.model = model
        
    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += "  === Material Parameters ===\n"
        s += f"  Name              : {self.get_property('name')}\n"
        s += f"  [Beam]  E           : {self.E}\n"
        s += f"  [Beam]  nu          : {self.nu}\n"
        s += f"  [Beam]  mu          : {self.mu}\n"
        s += f"  [Beam]  rho         : {self.rho}\n"
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
        
        L = bm.zeros((2, 2), dtype=bm.float64)
        
        L[0, 0] = 1 - xi
        L[0, 1] = xi
        L[1, 0] = -t
        L[1, 1] = t
        return L
    
    def hermite_basis(self, x: float, l: float, index: int = None) -> TensorLike:
        """Timoshenko beam shape functions considering shear deformation.

        Parameters:
            x (float): Local coordinate along the beam axis.
            l (float): Length of the beam element.
            index (int): Element index to get corresponding cross-section properties.

        Returns:
            h (TensorLike): Shape functions matrix, shape (3, 8).
                        Row 0: shape functions [N_v0, N_v1, N_v2, N_v3, N_w0, N_w1, N_w2, N_w3]
                        Row 1: first derivatives
                        Row 2: second derivatives
        """
        xi = x / l
        
        # 从 pde model 中获取截面性质
        if index is not None and hasattr(self.model, 'Ay') and hasattr(self.model, 'Az'):
            if isinstance(self.model.Ay, (list, tuple)) or hasattr(self.model.Ay, '__getitem__'):
                A_y = self.model.Ay[index]   # y方向有效剪切面积
                A_z = self.model.Az[index]   # z方向有效剪切面积
                I_yy = self.model.Iy[index]  # 绕y轴惯性矩
                I_zz = self.model.Iz[index]  # 绕z轴惯性矩
                mu_y = self.model.FSY       # 剪切修正系数
                mu_z = self.model.FSZ       # 剪切修正系数
            else:
                A_y = self.model.Ay
                A_z = self.model.Az
                I_yy = self.model.Iy
                I_zz = self.model.Iz
        mu_y = self.model.FSY       # 剪切修正系数
        mu_z = self.model.FSZ       # 剪切修正系数

        # 计算剪切变形参数
        Phi_y = 12 * self.E * I_yy / (mu_y * self.mu * A_z * l**2)
        Phi_z = 12 * self.E * I_zz / (mu_z * self.mu * A_y * l**2)

        Phi_bar_y = 1.0 / (1.0 + Phi_y)
        Phi_bar_z = 1.0 / (1.0 + Phi_z)
        
        H = bm.zeros((4, 8), dtype=bm.float64)
        
        # v方向的形函数 (考虑z方向弯曲)
        H[0, 0] = Phi_bar_z * (1 - 3*xi**2 + 2*xi**3 + Phi_z*(1 - xi))
        H[0, 1] = l * Phi_bar_z * (xi - 2*xi**2 + xi**3 + Phi_z*(xi - xi**2)/2)
        H[0, 2] = Phi_bar_z * (3*xi**2 - 2*xi**3 + Phi_z*xi)
        H[0, 3] = l * Phi_bar_z * (-xi**2 + xi**3 + Phi_z*(-xi + xi**2)/2)

        # w方向的形函数 (考虑y方向弯曲)
        H[0, 4] = Phi_bar_y * (1 - 3*xi**2 + 2*xi**3 + Phi_y*(1 - xi))
        H[0, 5] = -l * Phi_bar_y * (xi - 2*xi**2 + xi**3 + Phi_y*(xi - xi**2)/2)
        H[0, 6] = Phi_bar_y * (3*xi**2 - 2*xi**3 + Phi_y*xi)
        H[0, 7] = -l * Phi_bar_y * (-xi**2 + xi**3 + Phi_y*(-xi + xi**2)/2)
        
        # θy的形函数
        H[1, 0] = 6*Phi_bar_y * (-xi + 6*xi**2) / l
        H[1, 1] = -Phi_bar_y * (1 - 4*xi + 3*xi**2 + Phi_y*(1 - xi))
        H[1, 2] = -6*Phi_bar_y * (-xi + xi**2) / l
        H[1, 3] = -Phi_bar_y * (-2*xi + 3*xi**2 + Phi_y*xi)
        
        # θz的形函数
        H[1, 4] = 6*Phi_bar_z * (-xi + xi**2) / l
        H[1, 5] = Phi_bar_z * (1 - 4*xi + 3*xi**2 + Phi_z*(1 - xi))
        H[1, 6] = -6*Phi_bar_z * (-xi + xi**2) / l
        H[1, 7] = Phi_bar_z * (-2*xi + 3*xi**2 + Phi_z*xi)

        # 一阶导数(v、w方向)
        H[2, 0] = Phi_bar_z * (-6*xi + 6*xi**2 - Phi_z) / l
        H[2, 1] = Phi_bar_z * (1 - 4*xi + 3*xi**2 + Phi_z*(1 - 2*xi)/2)
        H[2, 2] = Phi_bar_z * (6*xi - 6*xi**2 + Phi_z) / l
        H[2, 3] = Phi_bar_z * (-2*xi + 3*xi**2 + Phi_z*(-1 + 2*xi)/2)

        H[2, 4] = Phi_bar_y * (-6*xi + 6*xi**2 - Phi_y) / l
        H[2, 5] = -Phi_bar_y * (1 - 4*xi + 3*xi**2 + Phi_y*(1 - 2*xi)/2)
        H[2, 6] = Phi_bar_y * (6*xi - 6*xi**2 + Phi_y) / l
        H[2, 7] =  -Phi_bar_y * (-2*xi + 3*xi**2 + Phi_y*(-1 + 2*xi)/2)

        # 一阶导数(θy、θz方向)
        H[3, 0] = 6*Phi_bar_y * (-1 + 12*xi) / l**2
        H[3, 1] = -Phi_bar_y * (-4 + 6*xi - Phi_y) / l
        H[3, 2] = -6*Phi_bar_y * (-1 + 2*xi) / l**2
        H[3, 3] = -Phi_bar_y * (-2 + 6*xi + Phi_y) / l
        
        H[3, 4] = 6*Phi_bar_z * (-1 + 2*xi) / l**2
        H[3, 5] = Phi_bar_z * (-4 + 6*xi - Phi_z) / l
        H[3, 6] = -6*Phi_bar_z * (-1 + 2*xi) / l**2
        H[3, 7] = Phi_bar_z * (-2 + 6*xi + Phi_z) / l

        return H
    
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
        G = self.mu
        kappa = 10/9
        
        D = bm.array([[E, 0, 0],
                      [0, G/kappa, 0],
                      [0, 0, G/kappa]], dtype=bm.float64)

        return D
    
    def compute_strain_and_stress(self,
                                mesh,
                                disp,
                                cross_section_coords=(0.0, 0.0),
                                axial_position=None,
                                coord_transform=None,
                                ele_indices=None) -> Tuple[TensorLike, TensorLike]:
        """Compute the beam element strain and stress.
            ε = B * u_e
            σ = D * ε
            
        Parameters:
            mesh: Mesh object containing beam elements.
            disp (TensorLike): Nodal displacement vector, shape (n_nodes * 6,).
            cross_section_coords (Tuple[float, float]): Local coordinates (y, z) in the beam cross-section.
            axial_position (Optional[float]): Evaluation position along the beam axis ∈ [0, L].
                If None, the value is evaluated at the element midpoint L/2
            coord_transform: Coordinate transformation matrices R, shape (NC, 12, 12).
        
        Returns:
            strain (TensorLike): Strain vectors [e_xx, e_xy, e_xz], shape (n_elements, 3).
            stress (TensorLike): Stress vectors [σ_xx, τ_xy, τ_xz], shape (n_elements, 3).
        """
        NC = mesh.number_of_cells()
        cells = mesh.entity('cell')
        lengths = mesh.entity_measure('cell')
        
        if ele_indices is None:
            ele_indices = range(NC)
            num_elements = NC
        else:
            num_elements = len(ele_indices)
            
        if coord_transform is None:
            raise ValueError("coord_transform must be provided.")
        R = coord_transform
    
        strain = bm.zeros((num_elements, 3))
        stress = bm.zeros((num_elements, 3))
        
        
        if axial_position is None:
            mid_x = lengths[ele_indices] / 2.0  # 单元中点
        else:
            mid_x = axial_position

        y, z = cross_section_coords  # 截面局部坐标

        for idx, i in enumerate(ele_indices):
            node_indices = cells[i]  # [node0_idx, node1_idx]
            element_disp = bm.concatenate([
                disp[node_indices[0]], 
                disp[node_indices[1]]
            ])  # shape: (12,)
            
            # 计算局部位移和转角
            local_disp = R[i] @ element_disp  # shape: (12,)
            u0 = local_disp[0:3]   # 节点0的位移
            a0 = local_disp[3:6]   # 节点0的转角
            u1 = local_disp[6:9]
            a1 = local_disp[9:12]
            
            L = self.linear_basis(mid_x[i], lengths[i])
            H = self.hermite_basis(mid_x[i], lengths[i], index=i)

            # 轴向应变: εxx = ∂u/∂x - y*∂θz/∂x + z*∂θy/∂x 
            e_xx = (u0[0]*L[1, 0] + u1[0]*L[1, 1] -
                    y*(u0[1]*H[3, 4] + a0[2]*H[3, 5] + u1[1]*H[3, 6] + a1[2]*H[3, 7]) +
                    z*(u0[2]*H[3, 0] + a0[1]*H[3, 1] + u1[2]*H[3, 2] + a1[1]*H[3, 3]))
            
            # xy平面剪切应变: γxy = ∂v/∂x - θz
            e_xy = ((u0[1]*H[2, 0] + a0[2]*H[2, 1] + u1[1]*H[2, 2] + a1[2]*H[2, 3]) - 
                    (u0[1]*H[1, 4] + a0[2]*H[1, 5] + u1[1]*H[1, 6] + a1[2]*H[1, 7]))

            # xz平面剪切应变: γxz = ∂w/∂x + θy
            e_xz = ((u0[2]*H[2, 4] + a0[1]*H[2, 5] + u1[2]*H[2, 6] + a1[1]*H[2, 7]) +
                     (u0[2]*H[1, 0] - a0[1]*H[1, 1] - u1[2]*H[1, 2] - a1[1]*H[1, 3]))
            
            strain[idx, 0] = e_xx
            strain[idx, 1] = e_xy
            strain[idx, 2] = e_xz
        
        stress = strain @ self.stress_matrix()
        
        return strain, stress