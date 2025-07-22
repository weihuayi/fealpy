from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_two_grid_lfem_2d_model import StationaryIncompressibleNSTwoGridLFEM2DModel
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import FromSympy

backend = 'numpy'
bm.set_backend(backend)
pde = FromSympy(mu=1, rho=100)
pde.select_pde["sinsinexp"]()

# 粗网格
n = 8
mesh = pde.init_mesh(nx=n, ny=n)
model = StationaryIncompressibleNSTwoGridLFEM2DModel(pde=pde)
model.method["Newton"]()
fem = model.fem
uH, pH =  model.run(tol=1e-10)
uh, ph = uH, pH
mesh.nodedata['uh'] = uh.reshape(2,-1).T
mesh.nodedata['ph'] = ph
mesh.to_vtk('test_H.vtu')

error = mesh.error(pde.velocity, uh)
print(error)
for i in range(5):
    # 函数插值
    uspace = fem.uspace
    pspace = fem.pspace
    usspace = uspace.scalar_space
    ucell2dof = usspace.cell_to_dof()
    u0 = uh.reshape(2,-1).T[...,0]
    u1 = uh.reshape(2,-1).T[...,1]
    u0c2f = u0[ucell2dof]
    u1c2f = u1[ucell2dof]
    pc2f = ph[pspace.cell_to_dof()]
    data = {'p':pc2f, 'u0':u0c2f,'u1':u1c2f}
    option = mesh.bisect_options(data=data, disp=False)

    mesh.bisect(None, option)
    fem.update_mesh(mesh)
    uspace = fem.uspace
    pspace = fem.pspace
    usspace = uspace.scalar_space
    
    ph = pspace.function()
    pcell2dof = pspace.cell_to_dof()
    ph[pcell2dof.reshape(-1)] = option['data']['p'].reshape(-1)
    
    ucell2dof = usspace.cell_to_dof()
    u0 = usspace.function()
    u0[ucell2dof.reshape(-1)] = option['data']['u0'].reshape(-1)
    u1 = usspace.function()
    u1[ucell2dof.reshape(-1)] = option['data']['u1'].reshape(-1)
    
    uh = uspace.function()
    uh[:] = bm.stack((u0[:],u1[:]), axis=1).T.flatten()

mesh.nodedata['uh'] = uh.reshape(2,-1).T
mesh.nodedata['ph'] = ph
error = mesh.error(pde.velocity, uh)
print(error)
mesh.to_vtk('test_h.vtu')

from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

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
A, F = model.lagrange_multiplier(A ,F)
x = spsolve(A, F, 'mumps') 

ugdof = fem.uspace.number_of_global_dofs()
uh_star = fem.uspace.function()
ph_star = fem.pspace.function()
uh_star[:] = x[:ugdof]
ph_star[:] = x[ugdof:-1]



print(mesh.error(pde.velocity, uh_star))








