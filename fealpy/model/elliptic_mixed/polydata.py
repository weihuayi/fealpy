import sympy as sp
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian, variantmethod
from fealpy.backend import TensorLike

class PolyData:
    def __init__(self):
        x1, x2 = sp.symbols('x1, x2', real=True)
        self.x1 = x1
        self.x2 = x2

        # 构造满足 Neumann 边界条件的解析解
        p1 = (x1 - 0.5)**2 * x1**2 * (1 - x1)**2
        p2 = (x2 - 0.5)**2 * x2**2 * (1 - x2)**2
        u_sym = p1 * p2
        self.u_sym = u_sym

        # 变量系数扩散张量
        A11 = 1 + x1**2
        A22 = 1 + x2**2

        # 梯度与通量
        grad_u = [sp.diff(u_sym, x1), sp.diff(u_sym, x2)]
        flux = [-A11 * grad_u[0], -A22 * grad_u[1]]

        div_flux = sp.diff(flux[0], x1) + sp.diff(flux[1], x2)

        # 反应项系数
        c = 2

        f_sym = -div_flux + c * u_sym
        self.f_sym = f_sym

        # 编译
        self._u_func = sp.lambdify((x1, x2), u_sym, 'numpy')
        self._grad_u_x1_func = sp.lambdify((x1, x2), grad_u[0], 'numpy')
        self._grad_u_x2_func = sp.lambdify((x1, x2), grad_u[1], 'numpy')
        self._flux1_func = sp.lambdify((x1, x2), flux[0], 'numpy')
        self._flux2_func = sp.lambdify((x1, x2), flux[1], 'numpy')
        self._f_func = sp.lambdify((x1, x2), f_sym, 'numpy')

    def domain(self):
        return [0., 1., 0., 1.]

    def geo_dimension(self) -> int:
        return 2

    @variantmethod('tri')
    def init_mesh(self, nx=10, ny=10):
        from fealpy.mesh import TriangleMesh
        return TriangleMesh.from_box(self.domain(), nx=nx, ny=ny)

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        return self._u_func(x1, x2)

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        gx = self._grad_u_x1_func(x1, x2)
        gy = self._grad_u_x2_func(x1, x2)
        return bm.stack([gx, gy], axis=-1)

    @cartesian
    def flux(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        q1 = self._flux1_func(x1, x2)
        q2 = self._flux2_func(x1, x2)
        return bm.stack([q1, q2], axis=-1)

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        shape = p.shape[:-1] + (2, 2)
        val = bm.zeros(shape)
        val[..., 0, 0] = 1 + x1**2
        val[..., 1, 1] = 1 + x2**2
        return val

    @cartesian
    def diffusion_coef_inv(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        shape = p.shape[:-1] + (2, 2)
        val = bm.zeros(shape)
        val[..., 0, 0] = 1 / (1 + x1**2)
        val[..., 1, 1] = 1 / (1 + x2**2)
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x1, x2 = p[..., 0], p[..., 1]
        return self._f_func(x1, x2)

    @cartesian
    def grad_dirichlet(self, p, space):
        return bm.zeros_like(p[..., 0])  # 不使用

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.zeros(p.shape[:-1], dtype=bool)  # 全为 Neumann 条件
