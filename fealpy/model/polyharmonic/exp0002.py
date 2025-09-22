from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import LshapeMesher

class Exp0002(LshapeMesher):
    """
    2D biharmonic problem on L-shaped domain:
    
    Δ²φ = h in Ω = [0,2]^2 \ [0,1]^2,
    with boundary conditions :
      ∂φ/∂n = h_1,  φ = h_2.
    
    Exact solution in polar coordinates (r, θ):
      φ(r, θ) = r^(3/2) * sin(3θ/2)
    
    Viscosity parameter nu = 0.01 (if needed for further modeling).
    """
    def __init__(self):
        super().__init__()

    def geo_dimension(self) -> int:
        """Return geometric dimension of the domain."""
        return 2
    
    def domain(self):
        """Return bounding box of the computational domain."""
        # Box large enough to contain L shape; actual domain excludes [-1,1]^2
        return self._domain
   
    def init_mesh(self, domain=(-1.0, 1.0, -1.0, 1.0), threshold=(0.0, 1.0, -1.0, 0.0), h=None, h_min=None):
        from ...mesh import TriangleMesh
        import gmsh
        # This requires a mesh generator that supports polygonal domains with holes.
        # Define L-shaped polygon as outer square minus inner square.

        self._domain = domain
        self._threshold = threshold

        xmin, xmax, ymin, ymax = self._domain
        x1, x2, y1, y2 = self._threshold
        # 小矩形四个角
        thresh_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        self.singular_points = None
        for (x, y) in thresh_corners:
            if xmin < x < xmax and ymin < y < ymax:
                # 只取在大矩形内部的 threshold 角点
                self.singular_points = (x, y)

        # 生成网格
        gmsh.initialize()
        gmsh.model.add("LShape")

        xmin, xmax, ymin, ymax = self.domain()
        x1, x2, y1, y2 = self._threshold

        # 建立大矩形
        big = gmsh.model.occ.addRectangle(xmin, ymin, 0, xmax - xmin, ymax - ymin)
        # 建立小矩形
        small = gmsh.model.occ.addRectangle(x1, y1, 0, x2 - x1, y2 - y1)
        # 差集 (L 形)
        domain_tag, _ = gmsh.model.occ.cut([(2, big)], [(2, small)])

        # 奇异点
        if self.singular_points is not None:
            sx, sy = self.singular_points
            sp = gmsh.model.occ.addPoint(sx, sy, 0)
            gmsh.model.occ.fragment(domain_tag, [(0, sp)])

        gmsh.model.occ.synchronize()

        # 设置整体网格尺寸
        if h is None:
            h = min((xmax - xmin) / 10, (ymax - ymin) / 10)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        if h_min is not None and self.singular_points is not None:
            # 在奇异点处设置更小的网格尺寸
            gmsh.model.mesh.setSize([(0, sp)], h_min)
        gmsh.model.mesh.generate(2)

        # 提取节点和单元
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(node_coords.reshape(-1, 3)[:, :2], dtype=bm.float64)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        # 取第一个单元类型
        if elem_node_tags:
            cells = bm.array(elem_node_tags[0], dtype=bm.int64).reshape(-1, 3) - 1  # 三角形
        else:
            cells = []

        gmsh.finalize()

        return TriangleMesh(nodes, cells)

    @cartesian
    def threshold(self, p):
        x = p[..., 0]
        y = p[..., 1]
        x1, x2, y1, y2 = self._threshold
        return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Exact solution φ(r, θ) = r^{3/2} sin(3θ/2)."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] +=2*bm.pi
        return r**(3/2) * bm.sin(3/2 * theta)
    
    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Gradient ∇φ in Cartesian coordinates."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2 * bm.pi
        sin = bm.sin
        cos = bm.cos
        u_x = 3*x*sin(3*theta/2)/(2*(x**2 + y**2)**(1/4)) - 3*y*cos(3*theta/2)/(2*(x**2 + y**2)**(1/4))
        u_y = 3*x*cos(3*theta/2)/(2*(x**2 + y**2)**(1/4)) + 3*y*sin(3*theta/2)/(2*(x**2 + y**2)**(1/4))
        u_x = bm.where(r>1e-14, u_x, 0)
        u_y = bm.where(r>1e-14, u_y, 0)
        return bm.stack((u_x, u_y), axis=-1)
    
    @cartesian
    def hessian(self, p: TensorLike) -> TensorLike:
        """Hessian matrix components of φ."""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt(x**2 + y**2)
        theta = bm.atan2(y, x)
        theta[theta<0] += 2 * bm.pi
        sin = bm.sin
        cos = bm.cos
        u_xx = 3*(x**2*sin(3*theta/2) - 2*x*y*cos(3*theta/2) - y**2*sin(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_xy = 3*(x**2*cos(3*theta/2) + 2*x*y*sin(3*theta/2) - y**2*cos(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_yy = 3*(-x**2*sin(3*theta/2) + 2*x*y*cos(3*theta/2) + y**2*sin(3*theta/2))/(4*(x**2 + y**2)**(5/4))
        u_xx = bm.where(r>1e-14, u_xx, 0)
        u_xy = bm.where(r>1e-14, u_xy, 0)
        u_yy = bm.where(r>1e-14, u_yy, 0)
        return bm.stack((u_xx, u_xy, u_yy), axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        return bm.zeros(p.shape[0])
    
    def get_flist(self) -> Sequence[TensorLike]:
        """Return list of functions for solution, gradient, hessian."""
        return [self.solution, self.gradient, self.hessian]
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition on ∂Ω: φ."""
        return self.solution(p)
    
    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        atol = 1e-12
        return (bm.abs(x ) < atol) | (bm.abs(x - 2.0) < atol) | \
               (bm.abs(y ) < atol) | (bm.abs(y - 2.0) < atol) | \
               ((x >= 1.0) & (x < 2.0) & (y > 1.0) & (y <= 2.0))  # cut-out region

