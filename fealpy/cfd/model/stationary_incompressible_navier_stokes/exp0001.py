from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d
import sympy as sp

class Exp0001(BoxMesher2d):
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.mesh = self.init_mesh(box = options.get('box', [0.0, 5.0, 0.0, 1.0]), 
                                   center = options.get('center', (1.0, 0.5)),
                                   radius = options.get('radius', 0.1),
                                   n_circle = options.get('n_circle', 60),
                                   h = options.get('h', 0.05))
        self._init_expr()
        super().__init__(box=self.box)

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def init_mesh(self, box, center, radius, n_circle=60, h=0.05):
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
        result[..., 0] = 4 * y * (1 - y)
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
            (bm.abs(x - 0.) < atol) 
        )
        return on_boundary

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(x - 5.) < atol) 
        )
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (
            (bm.abs(y - 0.) < atol) | (bm.abs(y - 1.) < atol)
        )
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

