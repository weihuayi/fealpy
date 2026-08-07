from fealpy.backend import backend_manager as bm
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
from fealpy.decorator import cartesian

options ={
    'backend': 'numpy',
    'pde': 3,
    'box':[-0.5, 2.7, -0.4, 0.4],
    'rho': 1.0,
    'mu': 0.001,
    'T0': 0.0,
    'T1': 4.0,
    'nt': 40000,
    'init_mesh': 'tri',
    'lc': 0.04,
    'method': 'IPCS',
    'solve': 'direct',
    'run': 'main'
}
bm.set_backend(options['backend'])
manager = CFDPDEModelManager('incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh()
model = IncompressibleNSLFEM2DModel(pde=pde, mesh = mesh, options = options)
print(type(model))
model.equation.set_constitutive(1)
model.equation.set_coefficient('viscosity', pde.mu)

mesh = model.mesh         
pde = model.pde
fem = model.fem
fem.dt = model.timeline.dt
maxstep = model.maxstep 
tol = model.tol 

u0 = fem.uspace.interpolate(cartesian(lambda p: pde.velocity(p, model.timeline.T0)))
p0 = fem.pspace.interpolate(cartesian(lambda p: pde.pressure(p, model.timeline.T0)))
print(type(u0))

mesh.nodedata['ph'] = p0
mesh.nodedata['uh'] = u0.reshape(model.mesh.GD,-1).T
mesh.to_vtk(f'air_foil_{str(0).zfill(10)}.vtu')
for i in range(model.timeline.NL-1):
    t  = model.timeline.current_time()
    model.logger.info(f"time={t}")
    
    u1,p1 = model.run['one_step'](u0, p0, maxstep, tol)
    u0[:] = u1
    p0[:] = p1

    mesh.nodedata['ph'] = p1
    mesh.nodedata['uh'] = u1.reshape(model.mesh.GD,-1).T
    mesh.to_vtk(f'air_foil_{str(i+1).zfill(10)}.vtu')

    model.timeline.advance()
