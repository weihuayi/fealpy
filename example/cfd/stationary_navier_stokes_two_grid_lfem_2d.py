from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_2d_model import StationaryIncompressibleNSLFEM2DModel
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import StationaryNSLFEMPolynomialPDE 
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import FromSympy


backend = 'numpy'
bm.set_backend(backend)
pde = FromSympy()
#pde = StationaryNSLFEMPolynomialPDE()
mesh = pde.set_mesh(64)
model = StationaryIncompressibleNSLFEM2DModel(pde=pde)
fem = model.fem # 控制空间次数以及积分
fem.set.uspace(p=3)
uh, ph =model.run(tol=1e-7)
 
