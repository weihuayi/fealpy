from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d
import sympy as sp

class Exp0002(BoxMesher2d):
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1e-3
        self.rho = 1.0
        self.mesh = self.init_mesh(box = options.get('box', [0.0, 2.2, 0.0, 0.41]), 
                                   centers = options.get('center', (0.2, 0.2)),
                                   radius = options.get('radius', 0.05),
                                   n_circle = options.get('n_circle', 1000),
                                   h = options.get('h', 0.005))
        self._init_expr()
        super().__init__(box=self.box)

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def init_mesh(self, box, centers, radius, n_circle=60, h=0.05):
        self.box = box
        self.centers = centers
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
            (box[0], box[3])]
        facets = [[0, 1], [1, 2], [2, 3], [3, 0]]

        hole_points = []
        for center in centers:
            cx, cy = center
            theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False)
            circle_pts = [(cx + radius*bm.cos(t), cy + radius*bm.sin(t)) for t in theta]
            offset = len(points)
            circle_facets = [[i + offset, (i + 1) % n_circle + offset] for i in range(n_circle)]

            points.extend(circle_pts)
            facets.extend(circle_facets)
            hole_points.append([cx, cy])  # 每个圆心都添加为空洞内点

        # 构建 meshpy 网格
        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets)
        mesh_info.set_holes(hole_points)

        mesh = build(mesh_info, max_volume=h**2)
        node = bm.array(mesh.points)
        cell = bm.array(mesh.elements)
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
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(x - 0.) < atol) &
            (y > 0.) & (y < 0.41))
        return on_boundary

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (bm.abs(x - 2.2) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(y - 0.) < atol) | (bm.abs(y - 0.41) < atol))
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
        
    @cartesian
    def velocity_gradient(self, p: TensorLike) -> TensorLike:
        pass
    
    @cartesian
    def pressure_gradient(self):
        pass

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        pass
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        pass

    def _init_expr(self):
        x, y = sp.symbols('x, y')
        u1 = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1)
        u2 = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2
        p = 10 * (2 * x - 1) * (2 * y - 1)
        mu = self.mu
        rho = self.rho
        u = sp.Matrix([u1, u2])
        gradu1x = u1.diff(x)
        gradu1y = u1.diff(y)
        gradu2x = u2.diff(x)
        gradu2y = u2.diff(y)

        # 不可压缩性
        assert sp.simplify(gradu1x + gradu2y) == 0  

        convection = sp.Matrix([u1 * gradu1x + u2 * gradu1y, u1 * gradu2x + u2 * gradu2y])
        diffusion = sp.Matrix([sp.diff(gradu1x, x) + sp.diff(gradu1y, y), sp.diff(gradu2x, x) + sp.diff(gradu2y, y)])
        gradp = sp.Matrix([p.diff(x), p.diff(y)])
        force = - mu*diffusion + rho*convection + gradp
        
        self.u = sp.lambdify([x, y], u, 'numpy')
        self.p = sp.lambdify([x, y], p, 'numpy')
        self.force = sp.lambdify([x, y], force, 'numpy')

