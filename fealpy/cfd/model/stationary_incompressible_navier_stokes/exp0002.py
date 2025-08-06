from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d
import sympy as sp

class Exp0002(BoxMesher2d):
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
        start_center = options['start_center']
        radius = options['radius']
        nx = options['nx']
        ny = options['ny']
        dx = options['dx']
        dy = options['dy']
        shift_angle = options['shift_angle']
        n_circle = options['n_circle']
        h = options['h']    

        self.box = box   
        self.radius = radius 
        from meshpy.triangle import MeshInfo, build
        from fealpy.mesh import TriangleMesh    
        # 矩形顶点
        points = [
        (box[0], box[2]),
        (box[1], box[2]),
        (box[1], box[3]),
        (box[0], box[3])]
        facets = [[0, 1], [1, 2], [2, 3], [3, 0]]
        hole_points = []

        # 构造圆柱中心阵列
        x0, y0 = start_center
        shift_rad = shift_angle * bm.pi / 180
        centers = []

        for i in range(nx):
            for j in range(ny):
                cx = x0 + i * dx
                cy = y0 + j * dy + i * dx * bm.tan(shift_rad)
                if box[0] + radius < cx < box[1] - radius and box[2] + radius < cy < box[3] - radius:
                    centers.append((cx, cy))

        # 构建所有圆形边界和空洞点
        for center in centers:
            cx, cy = center
            theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False)
            circle_pts = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]
            offset = len(points)
            circle_facets = [[i + offset, (i + 1) % n_circle + offset] for i in range(n_circle)]

            points.extend(circle_pts)
            facets.extend(circle_facets)
            hole_points.append([cx, cy])

        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets)
        mesh_info.set_holes(hole_points)

        mesh = build(mesh_info, max_volume=h**2)
        node = bm.array(mesh.points)
        cell = bm.array(mesh.elements)
        return TriangleMesh(node, cell)
    
    @cartesian
    def velocity_dirichlet(self, p:TensorLike) -> TensorLike:
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)

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
    def obstacle_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on obstacle."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.array(0.0)
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
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def is_pressure_boundary(self, p):
        is_out = self.is_outlet_boundary(p)
        return is_out
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (bm.abs(x - self.box[0]) < atol)
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
        radius = self.radius
        atol = 1e-12
        # 检查是否接近圆的边界
        on_boundary = bm.zeros_like(x, dtype=bool)
        for center in self.centers:
            cx, cy = center
            on_boundary |= bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
        return on_boundary
        