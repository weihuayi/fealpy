import matplotlib.pyplot as plt
from typing import List
from fealpy.backend import backend_manager as bm
from fealpy.fdm import LaplaceOperator, DirichletBC
from fealpy.mesh import UniformMesh
from fealpy.typing import TensorLike
from fealpy.solver import spsolve

bm.set_backend('numpy')

class SinSinPDEData2D:
    """
    2D Poisson problem:

        -Δu(x, y) = f(x, y),  (x, y) ∈ (0, 1) × (0, 1)
         u(x, y) = 0,         on ∂Ω

    with the exact solution:

        u(x, y) = sin(πx)·sin(πy)

    The corresponding source term is:

        f(x, y) = 2·π²·sin(πx)·sin(πy)

    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> List[float]:
        return [0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return bm.sin(pi * x) * bm.sin(pi * y)

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        du_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        return bm.stack([du_dx, du_dy], axis=-1)

    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return 2 * pi**2 * bm.sin(pi * x) * bm.sin(pi * y)

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)
    

pde_ell2d = SinSinPDEData2D()
domain = [0.0, 1.0, 0.0, 1.0]
extent = [0, 5, 0, 5]
mesh = UniformMesh(domain, extent)

L01 = LaplaceOperator(mesh)
A1 =L01.fast_assembly()

L01 = LaplaceOperator(mesh, method='fast')
A2 =L01()
maxit = 5
em = bm.zeros((3, maxit), dtype=bm.float64)
for i in range(maxit):
    L0 = LaplaceOperator(mesh)
    A = L0.assembly()
    f = mesh.interpolate(pde_ell2d.source).flatten()
    dbc = DirichletBC(mesh, pde_ell2d.dirichlet)
    A, f = dbc.apply(A, f)
    uh = mesh.function().flatten()
    uh[:] = spsolve(A, f, solver='scipy')
    em[0, i], em[1, i], em[2, i] = mesh.error(pde_ell2d.solution, uh)

    if i == maxit - 1:
        fig = plt.figure(4)
        axes = fig.add_subplot(111, projection='3d')
        NN = mesh.number_of_nodes()
        mesh.show_function(axes, uh.reshape(mesh.nx+1, mesh.ny+1))
        plt.title(f"Iteration {i+1}")
        plt.show()
    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])

# plt.figure(figsize=(10, 6))
# mesh.show_function(plt, uh)
# plt.show()
print("------------")



# A = L0.assembly()
# A1 = L0.fast_assembly()

# # method 和 call 方法只需要使用一种
# L01 = LaplaceOperator(mesh, method='fast')
# A1 =L01()

print("------------")