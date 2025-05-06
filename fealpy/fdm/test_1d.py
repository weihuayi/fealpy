import matplotlib.pyplot as plt
from typing import List
from fealpy.backend import backend_manager as bm
from fealpy.fdm import LaplaceOperator, DirichletBC
from fealpy.mesh import UniformMesh
from fealpy.typing import TensorLike
from fealpy.solver import spsolve

bm.set_backend('numpy')

class SinPDEData1D:
    """
    1D Poisson problem:

        -u''(x) = f(x),  x in (0, 1)
         u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(πx)

    The corresponding source term is:

        f(x) = π²·sin(πx)

    Dirichlet boundary conditions are applied at both ends of the interval.
    """

    def geo_dimension(self) -> int: 
        return 1

    def domain(self) -> List[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p
        pi = bm.pi
        val = bm.sin(pi * x)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p
        pi = bm.pi
        val = pi * bm.cos(pi * x)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        x = p
        pi = bm.pi
        val = pi**2 * bm.sin(pi * x)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)
    

pde_ell1d = SinPDEData1D()
domain = [0.0, 1.0]
extent = [0, 10]
mesh = UniformMesh(domain, extent)

maxit = 5
em = bm.zeros((3, maxit), dtype=bm.float64)
for i in range(maxit):
    L0 = LaplaceOperator(mesh)
    A = L0.assembly()
    f = mesh.interpolate(pde_ell1d.source).flatten()
    dbc = DirichletBC(mesh, pde_ell1d.dirichlet)
    A, f = dbc.apply(A, f)
    uh = mesh.function()
    uh[:] = spsolve(A, f, solver='scipy')
    em[0, i], em[1, i], em[2, i] = mesh.error(pde_ell1d.solution, uh)

    if i == maxit - 1:
        mesh.show_function(plt, uh)
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
# L01 = LaplaceOperator(mesh, method='fast_assembly')
# A1 =L01()

print("------------")