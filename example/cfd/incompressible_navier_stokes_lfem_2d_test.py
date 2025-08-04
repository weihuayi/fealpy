from fealpy.backend import backend_manager as bm
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
from fealpy.cfd.model.test.incompressible_navier_stokes.incompressible_navier_stokes_2d import FromSympy, NSLFEMChannelPDE
from fealpy.cfd.model.test.incompressible_navier_stokes.exp0001 import Exp0001
from fealpy.decorator import cartesian
from fealpy.fem import DirichletBC

pde = NSLFEMChannelPDE()
pde.set_mesh(nx=16, ny=16)
model = IncompressibleNSLFEM2DModel(pde)
model.method['IPCS']()

fem = model.fem
equation = model.equation
timeline = model.timeline
timeline.set_timeline(0, 10, 1000)
mesh = model.mesh

#model.equation.set_constitutive(2)
model.run(maxstep=10)


