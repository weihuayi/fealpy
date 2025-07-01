from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
import sympy as sp

class AcSinCosCosData2D:
    def __init__(self):
        self.box = [-1, 1, -1, 1]
        self.x, self.y, self.t = sp.symbols("x y t")
        self.gamma = 0.02
        self.eta = self.set_mesh()
        self.area = 4
        self.phi_expr = 2 + sp.sin(self.t) * sp.cos(sp.pi * self.x) * sp.cos(sp.pi * self.y)
        self.u1_expr = sp.pi * sp.sin(2 * sp.pi * self.y) * sp.sin(sp.pi * self.x) ** 2 * sp.sin(self.t)
        self.u2_expr = -sp.pi * sp.sin(2 * sp.pi * self.x) * sp.sin(sp.pi * self.y) ** 2 * sp.sin(self.t)

        self.phi = sp.lambdify((self.x, self.y, self.t), self.phi_expr, "numpy")
        self.u1 = sp.lambdify((self.x, self.y, self.t), self.u1_expr, "numpy")
        self.u2 = sp.lambdify((self.x, self.y, self.t), self.u2_expr, "numpy")
        self.init_force()

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2
    
    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box
    
    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 1.0]
    
    def init_force(self):
        x, y, t = self.x, self.y, self.t
        eta = self.eta
        gamma = self.gamma
        u1 = self.u1_expr
        u2 = self.u2_expr
        phi = self.phi_expr
        phi_t = sp.diff(phi, t)
        u_dot_grad_phi = u1 * sp.diff(phi, x) + u2 * sp.diff(phi, y)
        lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2)
        f_phi = (phi**3 - phi) / eta**2
        self.phi_force_expr = phi_t + u_dot_grad_phi - gamma * (lap_phi - f_phi)

    @cartesian
    def phi_force(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        """Return the force term for the phase field equation."""
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        phi_force = sp.lambdify((self.x, self.y, self.t), self.phi_force_expr, "numpy")
        val[:] = phi_force(x, y, t)
        return val
    
    @cartesian
    def velocity_field(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val = bm.set_at(val, (..., 0), self.u1(x, y, t))
        val = bm.set_at(val, (..., 1), self.u2(x, y, t))
        return val
    
    @cartesian
    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(x.shape, dtype=bm.float64)
        val = bm.set_at(val,(..., ), self.phi(x, y, t))
        return val
    
    @cartesian
    def init_solution(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        """Return the initial condition for the phase field."""
        return self.solution(p, t)
    
    def set_mesh(self, n=120):
        from ...mesh import TriangleMesh
        mesh = TriangleMesh.from_box(self.box, nx=n, ny=n)
        return 4* bm.min(mesh.entity_measure('edge'))
    
    def verify_phase_solution(self):
        x, y, t = self.x, self.y, self.t
        u1 = self.u1_expr
        u2 = self.u2_expr
        gamma = self.gamma
        eta = self.eta
        
        # 解析表达式
        phi = self.phi_expr
        phi_t = sp.diff(phi, t)
        lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2)
        f_phi = (phi**3 - phi) / eta**2
        u_dot_grad_phi = u1 * sp.diff(phi, x) + u2 * sp.diff(phi, y)
        phase_lhs = phi_t + u_dot_grad_phi
        phase_rhs = gamma * (lap_phi - f_phi) 
        phase_res = phase_lhs - phase_rhs - self.phi_force_expr
        print('相场方程残差:', sp.simplify(phase_res))