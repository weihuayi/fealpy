from fealpy.backend import backend_manager as bm

from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.exp0001 import Exp0001


#bm.set_backend('numpy')
#pde = FromSympy()
#pde.select_pde['cossin']()
pde = Exp0001()
mesh = pde.init_mesh()
#pde.select_pde['sinsinexp']()

model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh=mesh)
model.method['Newton']()
model.run['main']()

#mesh = pde.init_mesh(20,20)
#model.update_mesh(mesh)
#model.run()

