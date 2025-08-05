from fealpy.backend import backend_manager as bm

from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy

bm.set_backend('numpy')
pde = FromSympy()
pde.select_pde['cossin']()
#pde.select_pde['sinsinexp']()

model = StationaryIncompressibleNSLFEMModel(pde=pde)
model.method['Stokes']()
model.run['uniform_refine'](maxit=3)

#mesh = pde.init_mesh(20,20)
#model.update_mesh(mesh)
#model.run()

