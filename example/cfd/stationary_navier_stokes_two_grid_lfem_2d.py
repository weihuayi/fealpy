from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_two_grid_lfem_2d_model import StationaryIncompressibleNSTwoGridLFEM2DModel
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import StationaryNSLFEMPolynomialPDE 
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import FromSympy

backend = 'numpy'
bm.set_backend(backend)
pde = FromSympy(mu=1, rho=200)
pde.select_pde["sinsinexp"]()

# 粗网格
n = 4
mesh = pde.init_mesh(nx=n, ny=n)
model = StationaryIncompressibleNSTwoGridLFEM2DModel(pde=pde)
model.method["Newton"]()
fem = model.fem
uH, pH =  model.run(tol=1e-10)
mesh.nodedata['uH'] = uH
mesh.nodedata['pH'] = pH
mesh.to_vtk('test_H.vtu')

uh, ph = uH, pH
mesh.nodedata['uh'] = uh
#mesh.nodedata['ph'] = ph

for i in range(5):
    # 函数插值
    uspace = fem.uspace
    pspace = fem.pspace
    usspace = uspace.scalar_space
    ucell2dof = usspace.cell_to_dof()
    u0 = uh.reshape(mesh.GD,-1).T[...,0]
    u1 = uh.reshape(mesh.GD,-1).T[...,1]
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
    ucell2dof = usspace.cell_to_dof()
    ph[pcell2dof.reshape(-1)] = option['data']['p'].reshape(-1)
    
    u0 = usspace.function()
    u0[ucell2dof.reshape(-1)] = option['data']['u0'].reshape(-1)
    u1 = usspace.function()
    u1[ucell2dof.reshape(-1)] = option['data']['u1'].reshape(-1)
    uh = uspace.function()
    uh[:] = bm.stack((u0[:],u1[:]), axis=1).T.flatten()

mesh.to_vtk('test_h.vtu')













