from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.simulation.fem import BDF2
from fealpy.cfd.problem.IncompressibleNS import Channel

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace
from fealpy.solver import spsolve 
from fealpy.fem import DirichletBC


backend = 'pytorch'
#backend = 'numpy'
bm.set_backend(backend)

pde = Channel()
mesh = pde.set_mesh(64)
ns_eq = IncompressibleNS(pde)
fem = BDF2(ns_eq)
fem.dt = 0.001

u0 = fem.uspace.function()
u1 = fem.uspace.function()
u2 = fem.uspace.function()
p1 = fem.pspace.function()
p2 = fem.pspace.function()

BForm = fem.BForm()
LForm = fem.LForm()


BC = DirichletBC((fem.uspace,fem.pspace), gd=(pde.velocity, pde.pressure), 
                      threshold=(None, None), method='interp')


ugdof = fem.uspace.number_of_global_dofs()
output = './'
for i in range(1000):
    print("step:", i) 
    fem.update(u0, u1)
    A = BForm.assembly()
    b = LForm.assembly()
    A,b = BC.apply(A, b)

    x = spsolve(A, b, 'mumps')
    u2[:] = x[:ugdof]
    p2[:] = x[ugdof:]

    u0[:] = u1[:]
    u1[:] = u2[:]
    p1[:] = p2[:]

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    #mesh.to_vtk(fname=fname)

    uerror = mesh.error(pde.velocity, u2)
    perror = mesh.error(pde.pressure, p2)
    print("uerror:", uerror)
    print("perror:", perror)

