from typing import Optional, Tuple
from builtins import float, str

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial


class EulerBernoulliBeamMaterial(LinearElasticMaterial):
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
                I: Optional[float] = None,
                ) -> None:
        super().__init__(name=name, 
                        elastic_modulus= elastic_modulus, 
                        poisson_ratio=poisson_ratio,
                        shear_modulus=shear_modulus)

        self.E = self.get_property('elastic_modulus')
        self.nu = self.get_property('poisson_ratio')
        self.mu = self.get_property('shear_modulus')
        self.I = I

        self.L = model.L
        self.A = model.A
        self.f = model.f
        self.h = model.h
        self.l = model.l
        
        
    def __str__(self) -> str:
        """Return a multi-line summary including PDE type and key params."""
        return (
            f"\n  euler_bernoulli (2D Euler-Bernoulli PDE on  domain)\n"
            f"  Box dimensions: L = {self.L}\n"
            f"  young's modulus: E = {self.E}\n"
            f"  moment of inertia: I = {self.I}\n"
            f"  cross-sectional area: A = {self.A}\n"
            f"  distributed load: f = {self.f}\n"
            f"  section height: h = {self.h}\n"
        )


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
        
        h[1, 0] = 6*t0 * (t2 - t1)
        h[1, 1] = 1 - 4*t1 + 3*t2
        h[1, 2] = 6*t0 * (t1 - t2)
        h[1, 3] = 3*t2 - 2*t1

        h[2, 0] = t0**2 * (12*t1 - 6)
        h[2, 1] = t0 * (6*t1 - 4)
        h[2, 2] = t0**2 * (6 - 12*t1)
        h[2, 3] = t0 * (6*t1 - 2)
        return h

    def get_curvature_coeffs(self,v_i, theta_i, v_j, theta_j, l):
        """
        Get curvature coefficients A and B for an Euler-Bernoulli beam element.
        Parameters:
            v_i (float): Transverse displacement at node i.
            theta_i (float): Rotation at node i.
            v_j (float): Transverse displacement at node j.
            theta_j (float): Rotation at node j.
            l (float): Length of the beam element.
        Returns:
            A (float): Coefficient A in curvature expression.
            B (float): Coefficient B in curvature expression.
        """
        A = (6/l**2)*(2*v_i - 2*v_j) + (6/l)*(theta_i - theta_j)
        B = (6/l**2)*(-6*v_i + 6*v_j) + (6/l)*(-4*theta_i + 2*theta_j)
        return A, B
    
    def compute_strain(self, u):
        """
        Compute nodal strains for a beam composed of two 2-node Euler-Bernoulli elements (3 nodes total).

        Parameters:
            h (float): cross-section height.
            u (TensorLike or sequence): nodal DOFs [v1, th1, v2, th2, v3, th3], where vi are transverse
            displacements and thi are rotations for nodes 1, 2 and 3 respectively.

        Returns:
            tuple[TensorLike, TensorLike]: (top_strains, bottom_strains), each a 1-D array of length 3
            giving the strain at the top and bottom surface of nodes 1..3.
        """
        v1 = u[0]
        th1 = u[1]
        v2 = u[2]
        th2 = u[3]
        v3 = u[4]
        th3 = u[5]
        h = self.h
        y_top = h / 2
        y_bottom = -h / 2
        l = self.l
        L1 = l[0]  # 单元1长度
        L2 = l[1]  # 单元2长度
        # 单元1
        A1, B1 = self.get_curvature_coeffs(v1, th1, v2, th2, L1)
        
        # 外推到节点1 (xi=0) 和 节点2 (xi=1)
        kappa1_left = A1                    # xi = 0
        kappa1_right = A1 + B1              # xi = 1

        strain1_top_left = -y_top * kappa1_left
        strain1_bottom_left = -y_bottom * kappa1_left

        strain1_top_right = -y_top * kappa1_right
        strain1_bottom_right = -y_bottom * kappa1_right

        # 单元2
        A2, B2 = self.get_curvature_coeffs(v2, th2, v3, th3, L2)
        
        # 节点2 (xi=0) 和 节点3 (xi=1)
        kappa2_left = A2                    # xi = 0
        kappa2_right = A2 + B2              # xi = 1

        strain2_top_left = -y_top * kappa2_left
        strain2_bottom_left = -y_bottom * kappa2_left

        strain2_top_right = -y_top * kappa2_right
        strain2_bottom_right = -y_bottom * kappa2_right

        # 计算每个节点的应变
        strain_node1_top = strain1_top_left
        strain_node1_bottom = strain1_bottom_left
        # 节点2: 单元1右端 和 单元2左端 平均
        strain_node2_top = (strain1_top_right + strain2_top_left) / 2
        strain_node2_bottom = (strain1_bottom_right + strain2_bottom_left) / 2
        # 节点3: 只有单元2贡献
        strain_node3_top = strain2_top_right
        strain_node3_bottom = strain2_bottom_right
        # 返回每个节点的上下表面应变
        top_strains = bm.array([strain_node1_top, strain_node2_top, strain_node3_top])
        bottom_strains = bm.array([strain_node1_bottom, strain_node2_bottom, strain_node3_bottom])

        return top_strains, bottom_strains
    
    def compute_stress(self, strain_top: TensorLike, strain_bottom: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """Compute nodal stresses from nodal strains for a beam.

        Parameters:
            strain_top (TensorLike): 1-D array of length 3 giving the strain at the top surface of nodes 1..3.
            strain_bottom (TensorLike): 1-D array of length 3 giving the strain at the bottom surface of nodes 1..3.
        Returns:
            tuple[TensorLike, TensorLike]: (stress_top, stress_bottom), each a 1-D array of length 3
            giving the stress at the top and bottom surface of nodes 1..3.
        """
        E = self.E
        stress_top = E * strain_top
        stress_bottom = E * strain_bottom
        return stress_top, stress_bottom
    
    def get_extrapolation_matrix(self, l):
        """
        Get the extrapolation matrix for curvature from Gauss points to nodes.
        Parameters:
            l (float): Length of the beam element.
        Returns:
            E (TensorLike): Extrapolation matrix (2x2) from Gauss points to nodes.
        """
        a = 6 / l**2
        b = 6 / l
        
        # 2-point Gauss points in [0,1]
        xi1 = 0.5 * (1 - 1/bm.sqrt(3))  # ≈0.2113
        xi2 = 0.5 * (1 + 1/bm.sqrt(3))  # ≈0.7887
        
        # G: 高斯点处的曲率系数矩阵 (2x4)
        G = bm.array([
            [a*(2 - 6*xi1), b*(1 - 4*xi1), a*(-2 + 6*xi1), b*(-1 + 2*xi1)],
            [a*(2 - 6*xi2), b*(1 - 4*xi2), a*(-2 + 6*xi2), b*(-1 + 2*xi2)]
        ])
        
        # N: 节点处的曲率系数矩阵 (2x4)
        N = bm.array([
            [2*a,      b,      -2*a,     -b     ],  # kappa at node i (xi=0)
            [-4*a,    -3*b,     4*a,      b     ]   # kappa at node j (xi=1)
        ])
        
        # 外插矩阵
        E = N @ bm.linalg.inv(G)
        return E
    
    def get_extrapolation_kappa_g(self,v1, th1, v2, th2):
        """
        Get extrapolated curvatures at nodes from Gauss point curvatures.
        Parameters:
            v1 (float): Transverse displacement at node 1.
            th1 (float): Rotation at node 1.
            v2 (float): Transverse displacement at node 2.
            th2 (float): Rotation at node 2.
        Returns:
            kappa2_i (float): Extrapolated curvature at node i.
            kappa2_j (float): Extrapolated curvature at node j.
        """
        
        xi_g = bm.array([0.5 * (1 - 1/bm.sqrt(3)), 0.5 * (1 + 1/bm.sqrt(3))])  # 2-point Gauss points in [0,1]
        l=self.l[0]
        A , B= self.get_curvature_coeffs(v1, th1, v2, th2, l)
        kappa_g = A + B * xi_g
        # 在高斯点计算曲率
        xi_g = bm.array([0.5*(1-1/bm.sqrt(3)), 0.5*(1+1/bm.sqrt(3))])
        kappa_g = A + B * xi_g

        # 外推
        E = self.get_extrapolation_matrix(l)
        kappa2_i, kappa2_j = E @ kappa_g
        return kappa2_i, kappa2_j


