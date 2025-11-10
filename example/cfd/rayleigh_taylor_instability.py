from fealpy.backend import backend_manager as bm
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.cfd import CHNSFEMModel
import psutil

options ={
    'backend': 'pytorch',
    'pde': 1,
    'box':[-0.5, 2.7, -1.0, 1.0],
    'rho': 1.0,
    'lc': 0.04,
    'eps': 1e-10,
    'ns_method': 'BDF2',
    'ch_method': 'ch_fem',
    'solve': 'direct',
    'run': 'main'
}
bm.set_backend(options['backend'])
bm.set_default_device('cpu')
manager = CFDPDEModelManager('chann_hilliard_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh(64, 256)
model = CHNSFEMModel(pde=pde, mesh = mesh, options = options)
model.ns_equation.set_constitutive(2)
model.ns_equation.set_coefficient('viscosity', 1/pde.Re)
model.ns_equation.set_coefficient('pressure', 1)
model.ch_equation.set_coefficient('mobility', 1/pde.Pe)
model.ch_equation.set_coefficient('interface', pde.epsilon**2)
model.ch_equation.set_coefficient('free_energy', 1)

dt = 0.00125*bm.sqrt(bm.array(2))
ns_fem = model.ns_fem
ch_fem = model.ch_fem
ns_fem.dt = dt
ch_fem.dt = dt
phispace = ch_fem.space 

phi0 = ch_fem.space.interpolate(pde.init_interface)
phi1 = phispace.interpolate(pde.init_interface)
mu1 = phispace.function()
mu2 = phispace.function()

u0 = ns_fem.uspace.function()
u1 = ns_fem.uspace.function()
p1 = ns_fem.pspace.function()

mesh.nodedata['phi'] = phi1
mesh.nodedata['velocity'] = u1.reshape(2,-1).T  
fname = './' + 'test_'+ str(1).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

for i in range(1,2000):
    # 设置参数
    print("iteration:", i)
    print("内存占用",psutil.Process().memory_info().rss / 1024 ** 2, "MB")  # RSS内存(MB)  

    phi0, phi1, mu1, u0, u1, p1 = model.run['one_step'](phi0, phi1, mu1, u0, u1, p1)
       
    mesh.nodedata['phi'] = phi1
    mesh.nodedata['velocity'] = u1.reshape(2,-1).T  
    mesh.nodedata['pressure'] = p1 
    mesh.nodedata['rho'] = pde.rho
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)
