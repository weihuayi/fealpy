from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_2d_model import StationaryIncompressibleNSLFEM2DModel
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import StationaryNSLFEMPolynomialPDE, StationaryNSLFEMSinSinPDE 
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import StationaryNSTwoGridLFEMPDE 
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import FromSympy


backend = 'numpy'
bm.set_backend(backend)
pde = FromSympy()
#pde.select_pde['Polynomial']()
mesh = pde.set_mesh(128)
model = StationaryIncompressibleNSLFEM2DModel(pde=pde)
fem = model.fem # 控制空间次数以及积分
#u = fem.uspace.function()
#model.update(u0)
uh, ph =model.run()
print(mesh.error(uh, pde.velocity))
 
