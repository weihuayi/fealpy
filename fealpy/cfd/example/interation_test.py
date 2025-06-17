from fealpy.backend import backend_manager as bm
from fealpy.cfd.simulation.fem import Newton, Ossen
from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.problem.IncompressibleNS import Channel
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 

backend = 'numpy'
bm.set_backend(backend)

pde = Channel()
pde.set_mesh(8)
ns_eq = IncompressibleNS(pde)
#fem = Newton(ns_eq, pde.is_p_boundary)
fem = Ossen(ns_eq, pde.is_p_boundary)
fem.set.assembly(quadrature_order=5)

mesh = pde.mesh
fem.dt = 0.01
output = './'

u0 = fem.uspace.function()
p0 = fem.pspace.function()
u1 = fem.uspace.function()
p1 = fem.pspace.function()

BForm = fem.BForm()
LForm = fem.LForm()
ugdof = fem.uspace.number_of_global_dofs()
pgdof = fem.pspace.number_of_global_dofs()
mesh.nodedata['u'] = u1.reshape(2,-1).T
mesh.nodedata['p'] = p1

BC = DirichletBC((fem.uspace,fem.pspace), gd=(pde.velocity, pde.pressure), 
                     threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp')
#BC = DirichletBC((fem.uspace,fem.pspace), gd=(pde.velocity, pde.pressure), 
#                     threshold=(pde.is_u_boundary, None), method='interp')
#BC = DirichletBC((fem.uspace,fem.pspace), gd=(pde.velocity, pde.pressure), 
#                      threshold=(None, None), method='interp')

'''
for i in range(10):
    t = i*fem.dt
    print(f"第{i+1}步")
    print("time=", t)
    fem.update(u0, u0)
    
    A = BForm.assembly()
    b = LForm.assembly()
    #from fealpy.sparse import CSRTensor
    #B = A.to_scipy()
    #B[-pgdof:, -pgdof:] = 1e-8 * bm.eye(pgdof)
    #A = CSRTensor.from_scipy(B)
    A,b = BC.apply(A, b)

    x = spsolve(A, b, 'mumps')
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]

    u0[:] = u1
    p0[:] = p1
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    mesh.nodedata['uerror'] = fem.uspace.interpolate(pde.velocity)[:] - u1[:]
    mesh.nodedata['perror'] = fem.pspace.interpolate(pde.pressure)[:] - p1[:]
    mesh.to_vtk(fname=fname)

    #uerror = bm.max(bm.abs(fem.uspace.interpolate(pde.velocity)[:] - u1[:]))
    uerror = mesh.error(pde.velocity, u1)
    perror = mesh.error(pde.pressure, p1)
    print("uerror:", uerror)
    print("perror:", perror)
    #bcs = bm.array([[1/3,1/3,1/3]])
    #print(bm.mean(bm.einsum('cqii->cq',u1.grad_value(bcs))))

'''
for i in range(10):
    t = i*fem.dt
    print(f"第{i+1}步")
    print("time=", t)
     

    inneru_0 = fem.uspace.function()
    inneru_0[:] = u0[:]
    inneru_1 = fem.uspace.function()
    
    for j in range(10): 
        fem.update(inneru_0, u0)
        
        A = BForm.assembly()
        b = LForm.assembly()
        A,b = BC.apply(A, b)
    
        x = spsolve(A, b, 'mumps')
        inneru_1[:] = x[:ugdof]
        p1[:] = x[ugdof:]
        error = mesh.error(inneru_0, inneru_1)
        inneru_0[:] = inneru_1
        if error < 1e-12:
            print(error)
            u1[:] = inneru_1
            break

    u0[:] = u1
    p0[:] = p1
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    mesh.to_vtk(fname=fname)

    uerror = mesh.error(pde.velocity, u1)
    perror = mesh.error(pde.pressure, p1)
    print("uerror:", uerror)
    print("perror:", perror)
