
from fealpy.backend import bm
from fealpy.model import PDEDataManager
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from fealpy.unml import Field, Function, TestFunction, TrialFunction
from fealpy.unml import  dot, grad, dcell, dface, dedge, dnode


pde = PDEDataManager('poisson').get_example('coscos')

domain = pde.domain()

mesh = TriangleMesh.from_box(domain, nx=10, ny=10)
space = LagrangeFESpace(mesh, p=1)


u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dcell
b = pde.source * v * dcell


print(a)
print(b)
