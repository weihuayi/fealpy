from fealpy.backend import backend_manager as bm
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
from fealpy.cfd.model.test.incompressible_navier_stokes.incompressible_navier_stokes_2d import FromSympy, NSLFEMChannelPDE
from fealpy.cfd.model.test.incompressible_navier_stokes.exp0001 import Exp0001
from fealpy.decorator import cartesian
from fealpy.fem import DirichletBC


pde = FromSympy()
pde.select_pde['channel']()
#pde.select_pde['poly2d']()
#pde.select_pde['sinsincos']()
#
#pde = NSLFEMChannelPDE()

mesh = pde.set_mesh(5,5)
model = IncompressibleNSLFEM2DModel(pde)
model.equation.set_constitutive(2)

fem = model.fem
equation = model.equation
timeline = model.timeline
timeline.set_timeline(0, 10, 100)
#model.method['IPCS']()
model.method['Newton']()

#model.apply_bc.set('None')
u1,p = model.run(maxstep=1)
