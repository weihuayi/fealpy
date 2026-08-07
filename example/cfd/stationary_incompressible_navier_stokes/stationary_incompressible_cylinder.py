from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model import CFDPDEModelManager

options = {
    'backend': 'numpy',
    'pde': 1,
    'init_mesh': 'tri',
    'box': [0.0, 2.2, 0.0, 0.41],
    'center': (0.2, 0.2),
    'radius': 0.05,
    'n_circle': 1000,
    'lc': 0.004,
    'rho': 1.0,
    'mu': 1e-3,
    'method': 'Newton',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main_cylinder',
    'maxit': 1,
    'maxstep': 1000,
    'tol': 1e-10
}

bm.set_backend(options['backend'])
manager = CFDPDEModelManager('stationary_incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh()
model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh = mesh, options = options)
# model.__str__()


def benchmark(uh, ph):
    fem = model.fem
    mesh = model.mesh
    location = mesh.location
    ipoints = fem.uspace.interpolation_points()
    qf = mesh.quadrature_formula(q=4, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    vd = fem.uspace.function()
    vl = fem.uspace.function()
    vd[:len(ipoints)][pde.is_obstacle_boundary(ipoints)] = 1.0
    vl[len(ipoints):][pde.is_obstacle_boundary(ipoints)] = 1.0

    cellmeasure = model.mesh.entity_measure("cell")
    p = ph(bcs = bcs)
    grad_vd = fem.uspace.grad_value(uh = vd, bc = bcs)
    grad_uh = fem.uspace.grad_value(uh = uh, bc = bcs)
    cd = pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vd, cellmeasure) 
    cd += pde.rho * bm.einsum('n, knj, knij, kni, k -> ',ws, uh(bcs = bcs), 
                                                        grad_uh,
                                                        vd(bcs = bcs), cellmeasure)  
    cd -= bm.einsum('n, kn, knii, k -> ', ws, p, grad_vd, cellmeasure) 

    grad_vl = fem.uspace.grad_value(uh = vl, bc = bcs)
    cl = pde.mu * bm.einsum('n, knij, knij, k-> ', ws, grad_uh, grad_vl, cellmeasure)
    cl += pde.rho * bm.einsum('n, knj, knij, kni, k -> ', ws, 
                                                        uh(bcs = bcs), 
                                                        grad_uh, 
                                                        vl(bcs = bcs),
                                                        cellmeasure)
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

    cd = -500 * cd
    cl = -500 * cl
    delta_p = ph(bcs = bcs0, index = index0) - ph(bcs = bcs1, index = index1)
    return cd, cl, delta_p

def to_vtk(uh1, ph1):
    mesh.nodedata['ph'] = ph1
    mesh.nodedata['uh'] = uh1.reshape(2,-1).T
    mesh.to_vtk('stationary_2d.vtu')

from fealpy.decorator import cartesian

maxit = options['maxit']
maxstep = options['maxstep']
tol = options['tol']
cd = bm.zeros(maxit, dtype=bm.float64)
cl = bm.zeros(maxit, dtype=bm.float64)
delta_p = bm.zeros(maxit, dtype=bm.float64)
for i in range(maxit):
    print(f"number of cells: {mesh.number_of_cells()}")
    uh0 = model.fem.uspace.function()
    ph0 = model.fem.pspace.function()
    for j in range(maxstep):

        BForm, LForm = model.linear_system()  
        model.fem.update(uh0)
        A = BForm.assembly() 
        b = LForm.assembly()
        A, b = model.fem.apply_bc(A, b, pde)
        x = model.solve(A, b)

        ugdof = model.fem.uspace.number_of_global_dofs()
        uh1= model.fem.uspace.function()
        ph1 = model.fem.pspace.function()
        uh1[:] = x[:ugdof]
        ph1[:] = x[ugdof:]

        res_u = mesh.error(uh0, uh1)
        res_p = mesh.error(ph0, ph1)
        print(f"res_u: {res_u}, res_p: {res_p}")
        if res_u + res_p < tol:
            print(f"Converged at iteration {j+1}")
            break 
        uh0[:] = uh1
        ph0[:] = ph1
    cd[i], cl[i], delta_p[i] = benchmark(uh1, ph1)
    print(f"Drag coefficient: {cd[i]}, \nLift coefficient: {cl[i]}, \nPressure difference: {delta_p[i]}")
    # mesh.uniform_refine()
print(f"Drag coefficient: {cd}, \nLift coefficient: {cl}, \nPressure difference: {delta_p}")
to_vtk(uh1, ph1)





