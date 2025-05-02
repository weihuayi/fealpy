from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
from fealpy.ml.grad import gradient
from fealpy.typing import TensorLike
from typing import Callable, Dict

class Residual:
    _pde_registry: Dict[str, Callable] = {}
    _bc_registry: Dict[str, Callable] = {}

    def __init__(self, equation_type: str):
        self.equation_type = equation_type

    @classmethod
    def register_pde(cls, equation_type: str):
        """注册PDE残差计算的装饰器工厂"""
        def decorator(func: Callable):
            cls._pde_registry[equation_type] = func
            return func
        return decorator

    @classmethod
    def register_bc(cls, equation_type: str):
        """注册边界条件残差计算的装饰器工厂"""
        def decorator(func: Callable):
            cls._bc_registry[equation_type] = func
            return func
        return decorator

    def pde_residual(self, p: TensorLike, pde, s_real, s_imag) -> TensorLike:
        """计算PDE残差"""
        if self.equation_type not in self._pde_registry:
            raise NotImplementedError(
                f"PDE residual for {self.equation_type} not implemented")
        return self._pde_registry[self.equation_type](p, pde, s_real, s_imag)

    def bc_residual(self, p: TensorLike, pde, s_real, s_imag) -> TensorLike:
        """计算边界条件残差"""
        if self.equation_type not in self._bc_registry:
            raise NotImplementedError(
                f"BC residual for {self.equation_type} not implemented")
        return self._bc_registry[self.equation_type](p, pde, s_real, s_imag)

# Helmholtz方程的残差计算
@Residual.register_pde('helmholtz_2d')
def _helmholtz_pde_residual(p: TensorLike, pde, s_real, s_imag) -> TensorLike:
    u = s_real(p) + 1j * s_imag(p)
    f = pde.source(p)
    
    u_real = u.real
    u_imag = u.imag
    
    grad_u_real_x, grad_u_real_y = gradient(u_real, p, create_graph=True, split=True)
    grad_u_real_xx, _ = gradient(grad_u_real_x, p, create_graph=True, split=True)
    _, grad_u_real_yy = gradient(grad_u_real_y, p, create_graph=True, split=True)
    
    grad_u_imag_x, grad_u_imag_y = gradient(u_imag, p, create_graph=True, split=True)
    grad_u_imag_xx, _ = gradient(grad_u_imag_x, p, create_graph=True, split=True)
    _, grad_u_imag_yy = gradient(grad_u_imag_y, p, create_graph=True, split=True)
    
    u_xx = grad_u_real_xx + 1j * grad_u_imag_xx
    u_yy = grad_u_real_yy + 1j * grad_u_imag_yy
    
    return u_xx + u_yy + (pde.k**2) * u + f

@Residual.register_bc('helmholtz_2d')
def _helmholtz_bc_residual(p: TensorLike, pde, s_real, s_imag) -> TensorLike:
    u = s_real(p) + 1j * s_imag(p)
    x = p[..., 0]
    y = p[..., 1]
    
    n = bm.zeros_like(p)
    n[x > bm.abs(y), 0] = 1.0
    n[y > bm.abs(x), 1] = 1.0
    n[x < -bm.abs(y), 0] = -1.0
    n[y < -bm.abs(x), 1] = -1.0
    
    grad_u_real = gradient(u.real, p, create_graph=True, split=False)
    grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
    grad_u = grad_u_real + 1j * grad_u_imag
    
    kappa = bm.tensor(0.0 + 1j * pde.k)
    g = pde.robin(p=p, n=n)
    
    return (grad_u * n).sum(dim=-1, keepdim=True) + kappa * u - g

@Residual.register_pde('poisson_2d')
def _poisson_pde_residual(p: TensorLike, pde, s_real, s_imag) -> TensorLike:
    u = s_real(p)  # 实部网络
    f = pde.source(p)
    
    u_x, u_y = gradient(u, p, create_graph=True, split=True)
    u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
    _, u_yy = gradient(u_y, p, create_graph=True, split=True)
    
    return u_xx + u_yy + f

@Residual.register_bc('poisson_2d')
def _poisson_bc_residual(p: TensorLike, pde, s_real, s_imag) -> TensorLike:
    pass
