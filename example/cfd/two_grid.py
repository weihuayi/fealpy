from fealpy.backend import backend_manager as bm
from fealpy.cfd import StationaryIncompressibleNSLFEMModel
from fealpy.cfd import TwoGridModel
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy
from fealpy.cfd.equation import StationaryIncompressibleNS
import copy

pde = FromSympy(mu=1, rho=100)
pde.select_pde["sinsinexp"]()
equation = StationaryIncompressibleNS(pde)

fine_model = StationaryIncompressibleNSLFEMModel(pde)
coarsen_model = StationaryIncompressibleNSLFEMModel(pde)

mesh = coarsen_model.mesh
model = TwoGridModel(fine_model, coarsen_model)

uH, pH =  model.coarsen_model.run(tol=1e-10)
error_H = mesh.error(pde.velocity, uH)

uh, ph = model.refine_and_interpolate(5, uH, pH)
error_h = mesh.error(pde.velocity, uh)


from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

fem = coarsen_model.fem
BForm = fem.BForm()
LForm = fem.LForm()
fem.update(uh)
A = BForm.assembly() 
b = LForm.assembly()

BC = DirichletBC(
    (fem.uspace, fem.pspace), 
    gd=(pde.velocity, pde.pressure), 
    threshold=(pde.is_velocity_boundary, pde.is_pressure_boundary), 
    method='interp')

A, F = BC.apply(A, b)
A, F = coarsen_model.lagrange_multiplier(A ,F)
x = spsolve(A, F, 'mumps') 

ugdof = fem.uspace.number_of_global_dofs()
uh_star = fem.uspace.function()
ph_star = fem.pspace.function()
uh_star[:] = x[:ugdof]
ph_star[:] = x[ugdof:-1]



print("asdasda",mesh.error(pde.velocity, uh_star))


