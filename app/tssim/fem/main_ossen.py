#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: main_ossen.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Wed 15 Jan 2025 09:21:16 AM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.cfd import NSFEMSolver
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 
from fealpy.pde.navier_stokes_equation_2d import ChannelFlow
from fealpy.old.timeintegratoralg import UniformTimeLine

output = './'
T = 1
nt = 500
n = 16

pde = ChannelFlow()
mesh = pde.mesh(n)

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=1)
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

solver = NSFEMSolver(pde, mesh, pspace, uspace, dt, q=5)

u0 = uspace.function()
u1 = uspace.function()
p0 = pspace.function()
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['u'] = u1.reshape(2,-1).T
mesh.nodedata['p'] = p1
mesh.to_vtk(fname=fname)

BForm = solver.Ossen_BForm()
Lform = solver.Ossen_LForm()

BC = DirichletBC((uspace,pspace), gd=(pde.velocity, pde.pressure), 
                      threshold=(None, None), method='interp')
for i in range(100):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.Ossen_update(u0)
    A = BForm.assembly()
    b = Lform.assembly()
    A,b = BC.apply(A, b)

    x = spsolve(A, b, 'mumps')
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]

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

    timeline.advance()



