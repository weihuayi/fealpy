from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d
import sympy as sp

class Exp0001():
    def __init__(self, options: dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1e-3
        self.rho = 1.0

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.options['box']
    
    def init_mesh(self):
        options = self.options
        box = options['box']
        center = options['center']
        radius = options['radius']
        n_circle = options['n_circle'] 
        h = options['h']

        self.box = box
        self.center = center
        self.radius = radius
        self.n_circle = n_circle
        self.h = h
        from meshpy.triangle import MeshInfo, build
        from fealpy.mesh import TriangleMesh    
        # 矩形顶点
        points = [
            (box[0], box[2]),
            (box[1], box[2]),
            (box[1], box[3]),
            (box[0], box[3])
        ]

        # 矩形边界
        facets = [[0, 1], [1, 2], [2, 3], [3, 0]]

        # 圆的离散点
        cx, cy = center
        theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False)
        circle_points = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]

        # 圆的边界：顺时针编号（meshpy 要求空洞边界为顺时针）
        circle_facets = [[i, (i+1) % n_circle] for i in range(n_circle)]

        # 合并点和边界
        circle_offset = len(points)
        all_points = points + circle_points
        all_facets = facets + [[i[0]+circle_offset, i[1]+circle_offset] for i in circle_facets]

        # 设置空洞区域（空洞内一点）
        hole_point = [cx, cy]

        # meshpy 生成三角网格
        mesh_info = MeshInfo()
        mesh_info.set_points(all_points)
        mesh_info.set_facets(all_facets)
        mesh_info.set_holes([hole_point])  # 空洞位置

        mesh = build(mesh_info, max_volume=h**2)

        node = bm.array(mesh.points)
        cell = bm.array(mesh.elements)

        # 转为 FEALPy 的 TriangleMesh
        return TriangleMesh(node, cell)
        
    
    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 1.2 * y * (0.41 - y)/(0.41**2)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 1.2 * y * (0.41 - y)/(0.41**2)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.array(0.0)
        return result
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on wall."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def is_pressure_boundary(self, p):
        is_out = self.is_outlet_boundary(p)
        return is_out


    @cartesian
    def obstacle_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on obstacle."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)
   
    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        return result

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.0
        result[..., 1] = 0.0
        return result

    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(x - self.box[0]) < atol) &
            (y > self.box[2]) & (y < self.box[3]))
        return on_boundary

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (bm.abs(x - self.box[1]) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(y - self.box[2]) < atol) | (bm.abs(y - self.box[3]) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        cx, cy = self.center
        radius = self.radius
        atol = 1e-12
        # 检查是否接近圆的边界
        on_boundary = bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
        return on_boundary
        

