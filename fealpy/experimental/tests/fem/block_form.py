#!/usr/bin/python3
from fealpy.experimental.fem.form import Form
from fealpy.experimental.fem import BlockForm
from fealpy.experimental.fem import BilinearForm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import ScalarDiffusionIntegrator
from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
#bm.set_backend('pytorch')

mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=4, ny=4)
space = LagrangeFESpace(mesh, p=1)
space1 = LagrangeFESpace(mesh, p=2)

bform0 = BilinearForm(space)
bform0.add_integrator(ScalarDiffusionIntegrator())
bform1 = BilinearForm(space1)
bform1.add_integrator(ScalarDiffusionIntegrator())
#blockform = BlockForm([[bform0,None,None],[None,bform1,None],[None,None,bform0]])
blockform = BlockForm([[bform0,None],[None,bform1]])
#print(blockform.assembly())
u = bm.zeros(106)
print(blockform@u)
