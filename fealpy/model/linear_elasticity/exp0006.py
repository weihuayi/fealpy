<<<<<<< HEAD
from typing import Optional, List,Tuple,Callable
from ...mesh import TetrahedronMesh
from ...backend import backend_manager as bm
from ...decorator import cartesian, variantmethod
from ...typing import  TensorLike
from ...mesher import BoxMesher3d


class Exp0006():
    """
    -∇·σ = b    in Ω
      Aσ = ε(u) in Ω
       u = 0    on ∂Ω (homogeneous Dirichlet)
    where:
    - σ is the stress tensor
    - ε = (∇u + ∇u^T)/2 is the strain tensor
    - A is the compliance tensor
    Material parameters:
        lam = 1.0, mu = 0.5
    For isotropic materials:
        Aσ = (1/2μ)σ - (λ/(2μ(dλ+2μ)))tr(σ)I
    """
    def __init__(self, 
                domain: List[float] = [0, 1, 0, 1, 0, 1],
                mesh_type: str = 'uniform_tet'              
            ) -> None:
        self._eps = 1e-12
        self._plane_type = '3d'
        self._force_type = 'distribution'
        self._boundary_type = 'dirichlet'
        self._domain = domain
    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    def lam(self, p: Optional[TensorLike] = None) -> float:
        return 1.0

    def mu(self, p: Optional[TensorLike] = None) -> float:
        return 0.5
    
    @variantmethod('uniform_tet')
    def init_mesh(self, **kwargs) -> TetrahedronMesh:
        nx = kwargs.get('nx', 4)
        ny = kwargs.get('ny', 4)
        nz = kwargs.get('nz', 4)
        threshold = kwargs.get('threshold', None)
        device = kwargs.get('device', 'cpu')
        mesh = TetrahedronMesh.from_box(
                                    box=self._domain, 
                                    nx=nx, ny=ny, nz=nz,
                                    threshold=threshold, 
                                    device=device
                                )
        return mesh
    
    def stress_matrix_coefficient(self) -> tuple[float, float]:
        """
        材料为均匀各向同性线弹性体时, 计算应力块矩阵的系数 lambda0 和 lambda1
        Returns:
        --------
        lambda0: 1/(2μ)
        lambda1: λ/(2μ(dλ+2μ)), 其中 d=3 为空间维数
        """
        d = 3
        lambda0 = 1.0 / (2 * self.mu())
        lambda1 = self.lam() / (2 * self.mu() * (d * self.lam() + 2 * self.mu()))
        return lambda0, lambda1
    @cartesian
    def body_force(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        # 位移解的系数
        c1, c2, c3 = 16, 32, 64
        # 计算常用项
        xy = x * (1 - x) * y * (1 - y)
        xz = x * (1 - x) * z * (1 - z)
        yz = y * (1 - y) * z * (1 - z)
        # 计算导数项
        dx2_xy = (1 - 2*x) * (1 - 2*y)
        dx2_xz = (1 - 2*x) * (1 - 2*z)
        dy2_yz = (1 - 2*y) * (1 - 2*z)
        # 体力分量 b1
        b1_term1 = c1 * (xz + xy + 4*yz)
        b1_term2 = -3 * c1 * dx2_xy * z * (1 - z)
        b1_term3 = -6 * c1 * dx2_xz * y * (1 - y)
        b1 = b1_term1 + b1_term2 + b1_term3
        # 体力分量 b2
        b2_term1 = c2 * (yz + 4*xz + xy)
        b2_term2 = -0.75 * c2 * dx2_xy * z * (1 - z)
        b2_term3 = -3 * c2 * dy2_yz * x * (1 - x)
        b2 = b2_term1 + b2_term2 + b2_term3
        # 体力分量 b3
        b3_term1 = c3 * (yz + xz + 4*xy)
        b3_term2 = -0.375 * c3 * dx2_xz * y * (1 - y)
        b3_term3 = -0.75 * c3 * dy2_yz * x * (1 - x)
        b3 = b3_term1 + b3_term2 + b3_term3
        val = bm.stack([b1, b2, b3], axis=-1)
        return val
    @cartesian
    def disp_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        common = x * (1 - x) * y * (1 - y) * z * (1 - z)
        u1 = 16 * common
        u2 = 32 * common
        u3 = 64 * common
        val = bm.stack([u1, u2, u3], axis=-1)
        return val
    @cartesian
    def dirichlet_bc(self, points: TensorLike) -> TensorLike:
        val = self.disp_solution(points)
        return val
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps
        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1
        return flag
    @cartesian  
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps
        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1
        return flag
    @cartesian  
    def is_dirichlet_boundary_dof_z(self, points: TensorLike) -> TensorLike:
        domain = self.domain
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        flag_x0 = bm.abs(x - domain[0]) < self._eps
        flag_x1 = bm.abs(x - domain[1]) < self._eps
        flag_y0 = bm.abs(y - domain[2]) < self._eps
        flag_y1 = bm.abs(y - domain[3]) < self._eps
        flag_z0 = bm.abs(z - domain[4]) < self._eps
        flag_z1 = bm.abs(z - domain[5]) < self._eps
        flag = flag_x0 | flag_x1 | flag_y0 | flag_y1 | flag_z0 | flag_z1
        return flag
    def is_dirichlet_boundary(self) -> Tuple[Callable, Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y,
                self.is_dirichlet_boundary_dof_z)
    @cartesian
    def stress_solution(self, points: TensorLike) -> TensorLike:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        # 位移解的系数
        c1, c2, c3 = 16, 32, 64
        # 计算位移梯度
        # u1 = c1 * x(1-x)y(1-y)z(1-z)
        du1_dx = c1 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du1_dy = c1 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du1_dz = c1 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        # u2 = c2 * x(1-x)y(1-y)z(1-z)
        du2_dx = c2 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du2_dy = c2 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du2_dz = c2 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        # u3 = c3 * x(1-x)y(1-y)z(1-z)
        du3_dx = c3 * (1 - 2*x) * y * (1 - y) * z * (1 - z)
        du3_dy = c3 * x * (1 - x) * (1 - 2*y) * z * (1 - z)
        du3_dz = c3 * x * (1 - x) * y * (1 - y) * (1 - 2*z)
        # 计算应变张量 ε = (∇u + ∇u^T)/2
        eps_xx = du1_dx
        eps_yy = du2_dy
        eps_zz = du3_dz
        eps_xy = 0.5 * (du1_dy + du2_dx)
        eps_yz = 0.5 * (du2_dz + du3_dy)
        eps_xz = 0.5 * (du1_dz + du3_dx)
        # 计算应力张量 σ = λ(tr(ε))I + 2με
        lam, mu = self.lam(), self.mu()
        trace_eps = eps_xx + eps_yy + eps_zz
        sigma_xx = lam * trace_eps + 2 * mu * eps_xx
        sigma_yy = lam * trace_eps + 2 * mu * eps_yy
        sigma_zz = lam * trace_eps + 2 * mu * eps_zz
        sigma_xy = 2 * mu * eps_xy
        sigma_yz = 2 * mu * eps_yz
        sigma_xz = 2 * mu * eps_xz
        # 目前顺序: (σ_xx, σ_xz, σ_xy, σ_yy, σ_yz, σ_zz) - 与 HuzhangSpace 保持一致
        val = bm.stack([sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz], axis=-1)
        return val
    
=======
from typing import Optional, Sequence
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher2d



class Exp0006(BoxMesher2d):
    def __init__(self, box: Optional[Sequence[float]] = None):
        
        if box is None:
            box = [-1, 1, -1, 1]
        super().__init__(box)
        self.hypo = 'plane_strain'

    def lam(self, p: Optional[TensorLike] = None) -> float:
        return 1.0  

    def mu(self, p: Optional[TensorLike] = None) -> float:
        return 0.5  

    def stress_matrix_coefficient(self) -> tuple[float, float]:

        d = self.geo_dimension()
        lam, mu = self.lam(), self.mu()
        λ0 = 1.0 / (2 * mu)
        λ1 = lam / (2 * mu * (d * lam + 2 * mu))
        return λ0, λ1

    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.ones(tuple(shp), dtype=p.dtype)

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.zeros(tuple(shp), dtype=p.dtype)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[...,0], p[...,1]
        eps = 1e-12
        box = self.domain()  
        flag = (bm.abs(x-box[0])<eps) | (bm.abs(x-box[1])<eps) \
             | (bm.abs(y-box[2])<eps) | (bm.abs(y-box[3])<eps)
        return flag






>>>>>>> origin/develop
