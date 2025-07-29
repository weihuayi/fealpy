from fealpy.backend import backend_manager as bm
from fealpy.cfd import StationaryIncompressibleNSLFEMModel
from fealpy.cfd import TwoGridModel
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy
from fealpy.utils import timer

pde = FromSympy(mu=1, rho=1)
pde.select_pde["sinsin"]()
pde.init_mesh(nx=4, ny=4)
tmr = timer()
next(tmr)

coarsen_model = StationaryIncompressibleNSLFEMModel(pde)
fine_model = StationaryIncompressibleNSLFEMModel(pde)
mesh = coarsen_model.mesh
mesh.bisect()
model = TwoGridModel(fine_model, coarsen_model)
mesh.to_vtk('H.vtu')

tmr.send("粗网格求解开始")

#model.coarsen_model.method['Stokes']()
uH, pH =  model.coarsen_model.run(tol=1e-10)
error_H = mesh.error(pde.velocity, uH)
print("粗网格误差", error_H)
tmr.send("粗网格求解结束")

uh_refine, ph_refine = model.refine_and_interpolate(10, uH, pH)
model.fine_model.update_mesh(mesh)
mesh.to_vtk('h.vtu')
tmr.send("细网格插值结束")

uh, ph = model.fine_model.run(tol=1e-10)
print("细网格误差", mesh.error(pde.velocity, uh))
tmr.send("细网格求解结束")

uh_star, ph_star = model.fine_model.run['one_step'](uh_refine)
error_h = mesh.error(pde.velocity, uh_star)
print("两网格细网格误差", error_h)
tmr.send("细网格一步法结束")

uh_final, ph_final = model.correct_equation(uh_refine, uh_star)
error_final = mesh.error(pde.velocity, uh_final)
print("两网格细网格修正误差", error_final)
tmr.send("细网格修正结束")

next(tmr)
