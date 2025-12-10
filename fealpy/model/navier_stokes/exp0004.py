from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.mesher import BoxMesher2d

class Exp0004(BoxMesher2d):
    def __init__(self, options: dict = {}):
        self.options = options
        self.box = [0.0, 1.0, 0.0, 1.0]
        self.eps = 1e-10
        self.mu = 1.0
        self.rho = 1.0
        self.mesh = self.init_mesh(nx=options.get('nx', 8), ny=options.get('ny', 8))
        super().__init__(box=self.box)

    def __str__(self) -> str:
        """Return a nicely formatted, multi-line summary of the PDE configuration."""
        pass

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def set_mesh(self, nx, ny):
        mesh = super().init_mesh['uniform_tri'](nx=nx, ny=ny)
        self.mesh = mesh 
        return mesh
     
    @cartesian
    def velocity(self, p: TensorLike, t) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * bm.cos(t)
        result[..., 1] = -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * bm.cos(t)
        return result
    
    @cartesian
    def velocity_u(self, p: TensorLike, t) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return 10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * bm.cos(t)

    @cartesian
    def velocity_v(self, p: TensorLike, t) -> TensorLike:
        x, y = p[..., 0], p[..., 1]
        return -10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * bm.cos(t)

    @cartesian
    def velocity_0(self, p: TensorLike,t0):
        return self.velocity(p, t0)
    
    @cartesian
    def velocity_u0(self, p: TensorLike,t0):
        return self.velocity_u(p, t0)
    
    @cartesian
    def velocity_v0(self, p: TensorLike,t0):
        return self.velocity_v(p, t0)

    @cartesian
    def pressure_0(self, p: TensorLike,t0) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        return self.pressure(p, t0)
    
    @cartesian
    def pressure(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        return 10 * (2 * x - 1) * (2 * y - 1) * bm.cos(t)
    @cartesian
    def velocity_t(self, p: TensorLike, t) -> TensorLike:
        """时间导数 ∂u/∂t"""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = -10 * x ** 2 * (x - 1) ** 2 * y * (y - 1) * (2 * y - 1) * bm.sin(t)
        result[..., 1] = 10 * x * (x - 1) * (2 * x - 1) * y ** 2 * (y - 1) ** 2 * bm.sin(t)
        return result
    
    @cartesian
    def laplace_u(self, p: TensorLike, t) -> TensorLike:
        """拉普拉斯项 Δu"""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        
        # 计算 Δu1 = ∂²u1/∂x² + ∂²u1/∂y²
        # u1 = 10x²(x-1)²y(y-1)(2y-1)cos(t)
        
        # 计算 ∂²u1/∂x²
        term1_x = 240*x**2*y*(y-1)*(2*y-1)*bm.cos(t) - 240*x**3*y*(y-1)*(2*y-1)*bm.cos(t)
        term2_x = 80*x*y*(y-1)*(2*y-1)*bm.cos(t) - 240*x**2*y*(y-1)*(2*y-1)*bm.cos(t)
        term3_x = 20*y*(y-1)*(2*y-1)*bm.cos(t) - 80*x*y*(y-1)*(2*y-1)*bm.cos(t) + 60*x**2*y*(y-1)*(2*y-1)*bm.cos(t)
        
        # 计算 ∂²u1/∂y²
        term1_y = 10*x**2*(x-1)**2*(24*y-12)*bm.cos(t)
        term2_y = 10*x**2*(x-1)**2*(12*y**2-12*y+2)*bm.cos(t)
        
        d2u1_dx2 = 20*x**2*(x-1)**2*y*(y-1)*(2*y-1)*bm.cos(t) + 80*x*(2*x-2)*y*(y-1)*(2*y-1)*bm.cos(t) + 20*y*(x-1)**2*(y-1)*(2*y-1)*bm.cos(t)
        d2u1_dy2 = 10*x**2*(x-1)**2*(24*y-12)*bm.cos(t)
        
        # 简化计算
        result[..., 0] = 40*x**2*y*(x-1)**2*bm.cos(t) + 20*x**2*y*(y-1)*(2*y-1)*bm.cos(t) + 40*x**2*(x-1)**2*(y-1)*bm.cos(t) + 20*x**2*(x-1)**2*(2*y-1)*bm.cos(t) + 40*x*y*(2*x-2)*(y-1)*(2*y-1)*bm.cos(t) + 20*y*(x-1)**2*(y-1)*(2*y-1)*bm.cos(t)
        
        # 计算 Δu2 = ∂²u2/∂x² + ∂²u2/∂y²
        # u2 = -10x(x-1)(2x-1)y²(y-1)²cos(t)
        
        # 计算 ∂²u2/∂x²
        d2u2_dx2 = -10*y**2*(y-1)**2*(24*x-12)*bm.cos(t)
        
        # 计算 ∂²u2/∂y²
        d2u2_dy2 = -20*x*y**2*(x-1)*(2*x-1)*bm.cos(t) - 40*x*y**2*(y-1)**2*bm.cos(t) - 40*x*y*(x-1)*(2*x-1)*(2*y-2)*bm.cos(t) - 20*x*(x-1)*(2*x-1)*(y-1)**2*bm.cos(t) - 40*y**2*(x-1)*(y-1)**2*bm.cos(t) - 20*y**2*(2*x-1)*(y-1)**2*bm.cos(t)
        
        result[..., 1] = -20*x*y**2*(x-1)*(2*x-1)*bm.cos(t) - 40*x*y**2*(y-1)**2*bm.cos(t) - 40*x*y*(x-1)*(2*x-1)*(2*y-2)*bm.cos(t) - 20*x*(x-1)*(2*x-1)*(y-1)**2*bm.cos(t) - 40*y**2*(x-1)*(y-1)**2*bm.cos(t) - 20*y**2*(2*x-1)*(y-1)**2*bm.cos(t) - 10*y**2*(y-1)**2*(24*x-12)*bm.cos(t)
        
        return result
    
    @cartesian
    def pressure_grad(self, p: TensorLike, t) -> TensorLike:
        """压力梯度 ∇p"""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        # p = 10(2x-1)(2y-1)cos(t)
        result[..., 0] = 20 * (2*y - 1) * bm.cos(t)  # ∂p/∂x
        result[..., 1] = 20 * (2*x - 1) * bm.cos(t)  # ∂p/∂y
        return result
    
    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        """Compute exact source: f = ∂u/∂t - Δu + ∇p"""
        # 获取各个项
        u_t = self.velocity_t(p, t)  # ∂u/∂t
        laplace_u = self.laplace_u(p, t)  # Δu
        grad_p = self.pressure_grad(p, t)  # ∇p
        
        # 计算源项: f = ∂u/∂t - Δu + ∇p
        result = u_t - laplace_u + grad_p
        
        return result
    # @cartesian
    # def source(self, p: TensorLike, t) -> TensorLike:
    #     """Compute exact source """
    #     x = p[..., 0]
    #     y = p[..., 1]
    #     result = bm.zeros(p.shape, dtype=bm.float64)
    #     result[..., 0] = 10.0*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(10*x**2*y*(2*x - 2)*(y - 1)*(2*y - 1)*bm.cos(t) + 20*x*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t))*bm.cos(t) - 10*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.sin(t) - 40.0*x**2*y*(x - 1)**2*bm.cos(t) - 20.0*x**2*y*(y - 1)*(2*y - 1)*bm.cos(t) - 40.0*x**2*(x - 1)**2*(y - 1)*bm.cos(t) - 20.0*x**2*(x - 1)**2*(2*y - 1)*bm.cos(t) - 10.0*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(20*x**2*y*(x - 1)**2*(y - 1)*bm.cos(t) + 10*x**2*y*(x - 1)**2*(2*y - 1)*bm.cos(t) + 10*x**2*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t))*bm.cos(t) - 40.0*x*y*(2*x - 2)*(y - 1)*(2*y - 1)*bm.cos(t) - 20.0*y*(x - 1)**2*(y - 1)*(2*y - 1)*bm.cos(t) + 20*(2*y - 1)*bm.cos(t)
    #     result[..., 1] = 10.0*x**2*y*(x - 1)**2*(y - 1)*(2*y - 1)*(-20*x*y**2*(x - 1)*(y - 1)**2*bm.cos(t) - 10*x*y**2*(2*x - 1)*(y - 1)**2*bm.cos(t) - 10*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t))*bm.cos(t) - 10.0*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*(-10*x*y**2*(x - 1)*(2*x - 1)*(2*y - 2)*bm.cos(t) - 20*x*y*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t))*bm.cos(t) + 10*x*y**2*(x - 1)*(2*x - 1)*(y - 1)**2*bm.sin(t) + 20.0*x*y**2*(x - 1)*(2*x - 1)*bm.cos(t) + 40.0*x*y**2*(y - 1)**2*bm.cos(t) + 40.0*x*y*(x - 1)*(2*x - 1)*(2*y - 2)*bm.cos(t) + 20.0*x*(x - 1)*(2*x - 1)*(y - 1)**2*bm.cos(t) + 40.0*y**2*(x - 1)*(y - 1)**2*bm.cos(t) + 20.0*y**2*(2*x - 1)*(y - 1)**2*bm.cos(t) + 2*(20*x - 10)*bm.cos(t)
    #     return result
    
    @cartesian
    def source_u(self, p: TensorLike, t) -> TensorLike:
        
        return self.source(p,t)[..., 0]

    @cartesian
    def source_v(self, p: TensorLike, t) -> TensorLike:
        
        return self.source(p,t)[..., 1]

    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        # result = bm.ones_like(p[..., 0], dtype=bm.bool)
        # return result
        return None

    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return 0

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        return result
    
    
    @cartesian
    def velocity_dirichlet_u(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        # y = p[..., 1]
        result = bm.zeros(x.shape, dtype=bm.float64)
        return result
    @cartesian
    def velocity_dirichlet_v(self, p: TensorLike) -> TensorLike:
        # x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(y.shape, dtype=bm.float64)
        return result

    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        return None
