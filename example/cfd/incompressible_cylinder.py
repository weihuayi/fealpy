from fealpy.backend import backend_manager as bm
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
import argparse

# 解析参数
options ={
    'backend': 'numpy',
    'pde': 2,
    'rho': 1.0,
    'mu': 0.001,
    'T0': 0.0,
    'T1': 6.0,
    'nt': 6000,
    'init_mesh': 'tri',
    'box': [0.0, 2.2, 0.0, 0.41],
    'center': (0.2, 0.2),
    'radius': 0.05,
    'n_circle': 300,
    'lc': 0.05,
    'method': 'IPCS',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main_cylinder',
    'maxit': 5,
    'maxstep': 10,
    'tol': 1e-10
}

bm.set_backend(options['backend'])
# bm.set_default_device('cpu')
manager = CFDPDEModelManager('incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh()
model = IncompressibleNSLFEM2DModel(pde=pde, mesh = mesh, options = options)
model.equation.set_constitutive(1)
model.equation.set_coefficient('viscosity', pde.mu)

mesh = model.mesh         
pde = model.pde
fem = model.fem
fem.dt = model.timeline.dt
maxstep = model.maxstep 
tol = model.tol 

def benchmark(uh, ph, uh0):
    location = mesh.location
    qf = mesh.quadrature_formula(q=4, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    vd = fem.uspace.function()
    ipoints = fem.uspace.interpolation_points()
    vd[:len(ipoints)][model.pde.is_obstacle_boundary(ipoints)] = 1.0
    vl = fem.uspace.function()
    vl[len(ipoints):][model.pde.is_obstacle_boundary(ipoints)] = 1.0

    cellmeasure = model.mesh.entity_measure("cell")
    p = ph(bcs = bcs)
    u = uh(bcs = bcs)
    u0 = uh0(bcs = bcs)
    grad_vd = model.fem.uspace.grad_value(uh = vd, 
                                            bc = bcs)
    grad_vl = model.fem.uspace.grad_value(uh = vl, 
                                            bc = bcs)
    grad_uh = model.fem.uspace.grad_value(uh = uh, 
                                            bc = bcs)
    
    cd = (1/fem.dt) * bm.einsum('n, kni, kni, k -> ', ws, u, vd(bcs = bcs), cellmeasure)
    cd -= (1/fem.dt) * bm.einsum('n, kni, kni, k -> ', ws, u0, vd(bcs = bcs), cellmeasure)
    cd += model.pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vd, cellmeasure) 
    cd += model.pde.rho * bm.einsum('n, knj, knij, kni, k -> ',ws, uh(bcs = bcs), 
                                                grad_uh,
                                                vd(bcs = bcs), cellmeasure)  
    cd -= bm.einsum('n, knii, kn, k -> ', ws, grad_vd, p, cellmeasure) 

    cl = (1/fem.dt) * bm.einsum('n, kni, kni, k -> ', ws, u, vl(bcs = bcs), cellmeasure)
    cl -= (1/fem.dt) * bm.einsum('n, kni, kni, k -> ', ws, u0, vl(bcs = bcs), cellmeasure)
    cl += model.pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vl, cellmeasure) 
    cl += model.pde.rho * bm.einsum('n, knj, knij, kni, k -> ',ws, uh(bcs = bcs), 
                                                grad_uh,
                                                vl(bcs = bcs), cellmeasure)  
    cl -= bm.einsum('n, knii, kn, k -> ', ws, grad_vl, p, cellmeasure)  

    point0 = bm.array([[0.15, 0.2]])
    point1 = bm.array([[0.25, 0.2]])
    index0 = location(points=point0)
    index1 = location(points=point1)

    def get_bcs(point, index):
        node_points = mesh.entity("node")
        c2n = mesh.cell_to_node()
        cell = node_points[c2n][index][0]
        S = 0.5 * bm.cross(cell[1]-cell[0], cell[2]-cell[0])
        lambda1 = 0.5 * bm.cross(cell[1]-point[0], cell[2]-point[0]) / S
        lambda2 = 0.5 * bm.cross(cell[2]-point[0], cell[0]-point[0]) / S
        lambda3 = 1.0 - lambda1 - lambda2
        bcs = bm.array([[lambda1, lambda2, lambda3]])
        return bcs
    
    bcs0 = get_bcs(point=point0, index = index0)
    bcs1 = get_bcs(point=point1, index = index1)

    cd = -20 * cd
    cl = -20 * cl
    delta_p = ph(bcs = bcs0, index = index0) - ph(bcs = bcs1, index = index1)
    return cd, cl, delta_p



from fealpy.decorator import cartesian
# import vtk

# reader = vtk.vtkXMLUnstructuredGridReader()
# reader.SetFileName("/home/libz/Bob FEALPY/ns2d_0000014370.vtu")
# reader.Update()

# ugrid = reader.GetOutput()
# points = ugrid.GetPoints().GetData()
# print(points)

# exit()
u0 = fem.uspace.interpolate(cartesian(lambda p: pde.velocity(p, model.timeline.T0)))
p0 = fem.pspace.interpolate(cartesian(lambda p: pde.pressure(p, model.timeline.T0)))
cd = bm.zeros(model.timeline.NL-1)
cl = bm.zeros(model.timeline.NL-1)
delta_p = bm.zeros(model.timeline.NL-1)

mesh.nodedata['ph'] = p0
mesh.nodedata['uh'] = u0.reshape(model.mesh.GD,-1).T
mesh.to_vtk(f'ns2d_{str(0).zfill(10)}.vtu')
for i in range(model.timeline.NL-1):
    t  = model.timeline.current_time()
    model.logger.info(f"time={t}")
    
    u1,p1 = model.run['one_step'](u0, p0, maxstep, tol)
    
    cd[i], cl[i], delta_p[i] = benchmark(u1, p1, u0)
    print(f"Drag coefficient: {cd[i]}, \nLift coefficient: {cl[i]}, \nPressure difference: {delta_p[i]}")
    u0[:] = u1
    p0[:] = p1

    if i < 20000 :
        mesh.nodedata['ph'] = p1
        mesh.nodedata['uh'] = u1.reshape(model.mesh.GD,-1).T
        mesh.to_vtk(f'ns2d_{str(i+1).zfill(10)}.vtu')

    model.timeline.advance()


# uh, ph = model.run()
# cd = model.cd
# cl = model.cl
# delta_p = model.delta_p
# x = bm.linspace(0.0, 6.0, model.timeline.NL)
# model.__str__()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(x[16000:], cd[15999:], marker=None, linestyle='-', color='black')
# plt.xscale('linear')
# plt.yscale('linear')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Drag coefficient', fontsize=14)
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(x[16000:], cl[15999:], marker=None, linestyle='-', color='black')
# plt.xscale('linear')
# plt.yscale('linear')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Lift coefficient', fontsize=14)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(x[16000:], delta_p[15999:], marker=None, linestyle='-', color='black')
# plt.xscale('linear')
# plt.yscale('linear')
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Pressure difference', fontsize=14)
# plt.show()
