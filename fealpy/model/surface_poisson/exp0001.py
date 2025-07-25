from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...typing import  TensorLike
from ...mesher import SphereSurfaceMesher


class Exp0001(SphereSurfaceMesher):

    """
    Surface Poisson problem on a closed manifold:

        -Δ_S u(x, y, z) = f(x, y, z),   (x, y, z) ∈ S

    where:
        - S ⊂ ℝ³ is a closed, compact, C³-smooth surface without boundary (∂S = ∅),
        - Δ_S is the Laplace-Beltrami operator on surface S,
        - f is a given function defined on S, satisfying the compatibility condition:
              ∫_S f dσ = 0,
          where dσ is the surface measure on S,
        - To ensure uniqueness of the solution u, an additional constraint is imposed:
              ∫_S u dσ = 0.

    In this example:
        - The surface S is the unit sphere, represented implicitly as the level set:
              x² + y² + z² - 1 = 0.
        - The exact solution is chosen as:
              u(x, y, z) = x·y.
        - The corresponding source term f is defined as:
              f = -Δ_S (x·y).
    """

    def __init__(self):
        super().__init__()
        
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return x*y
    
    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        denom = x**2 + y**2 + z**2
        scale = 2*x*y / denom

        grad_x = y - scale * x
        grad_y = x - scale * y
        grad_z = -scale * z

        return bm.stack([grad_x, grad_y, grad_z], axis=-1)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        denom = x**2 + y**2 + z**2
        return 6*x*y / denom
    
    @cartesian
    def neumann(self, p: TensorLike, n: int) -> TensorLike:
        """
        Compute Neumann boundary condition.

        p: (NF, NQ, 3)
        n: (NF, 3)
        grad*n: (NQ, NE, 3)
        """
        grad = self.gradient(p) # (NF, NQ, 3)
        if n.ndim == 2:
            n = bm.expand_dims(n, axis=1)
        val = bm.einsum('fqd, fqd -> fq', grad, n) # (NF, NQ)
        return val