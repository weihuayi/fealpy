from fealpy.backend import backend_manager as bm
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
from fealpy.cfd.model.test.incompressible_navier_stokes.incompressible_navier_stokes_2d import FromSympy
from fealpy.cfd.model.incompressible_navier_stokes.poiseuille_2d import Poiseuille2D


pde = FromSympy()
#pde.select_pde['poly2d']()
pde.select_pde['polycos']()
#pde.select_pde['sinsincos']()
#pde.select_pde['channel']()

mesh = pde.init_mesh['uniform_tri'](10, 10)
model = IncompressibleNSLFEM2DModel(pde, mesh=mesh)
model.equation.set_constitutive(2)

fem = model.fem
equation = model.equation
timeline = model.timeline
timeline.set_timeline(0, 1, 500)
#model.method['IPCS']()
model.method['Newton']()

u1,p = model.run(maxstep=1)
mesh = model.mesh

mesh.nodedata['p'] = p
mesh.nodedata['u1'] = u1.reshape(2,-1).T
mesh.to_vtk('ns2d.vtu')
