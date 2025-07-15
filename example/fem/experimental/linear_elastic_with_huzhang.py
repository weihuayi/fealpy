
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
#from fealpy.fem.huzhang_displacement_integrator import HuZhangDisplacementIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator

from fealpy.decorator import cartesian

from fealpy.fem import BilinearForm,ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator,BoundaryFaceSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import BlockForm,LinearBlockForm

from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from fealpy.solver import spsolve
from scipy.sparse import csr_matrix, coo_matrix, bmat

import sys
import time


def solve(pde, N, p):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    space0 = HuZhangFESpace2d(mesh, p=p)

    space = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space1 = TensorFunctionSpace(space, shape=(-1, 2))

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1))

    bform2 = BilinearForm((space1,space0))
    bform2.add_integrator(HuZhangMixIntegrator())

    A = BlockForm([[bform1,bform2],
                   [bform2.T,None]])
    A = A.assembly()

    lform1 = LinearForm(space1)

    @cartesian
    def source(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=source))

    b = lform1.assembly()
    #a = displacement_boundary_condition(space0, pde.displacement)

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    #F[:gdof0] = a
    F[gdof0:] = -b

    X = spsolve(A, F, "scipy")

    sigmaval = X[:gdof0]
    uval = X[gdof0:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    uh = space1.function()
    uh[:] = uval
    return sigmah, uh


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 5
    p = int(sys.argv[1])

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    x, y = symbols('x y')

    pi = bm.pi 
    u0 = (sin(pi*x)*sin(pi*y))**2
    u1 = (sin(pi*x)*sin(pi*y))**2
    #u0 = sin(5*x)*sin(7*y)
    #u1 = cos(5*x)*cos(4*y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, lambda0, lambda1)

    for i in range(maxit):
        N = 2**(i+1) 
        sigmah, uh = solve(pde, N, p)
        mesh = sigmah.space.mesh

        e0 = mesh.error(uh, pde.displacement) 
        e1 = mesh.error(sigmah, pde.stress)

        h[i] = 1/N
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0 
        print(N, e0, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























