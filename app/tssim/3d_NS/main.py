#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: main.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Wed 04 Dec 2024 05:19:28 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.pde.navier_stokes_equation_2d import ChannelFlow
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.cfd import NSFEMSolver
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 

output = './'
h = 1/8
T = 1
nt = int(T/(0.1*h))

pde = ChannelFlow()
mesh = pde.mesh(int(1/h))


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=1)
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))

solver = NSFEMSolver(pde, mesh, pspace, uspace, dt, q=5)

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

u1 = uspace.function()
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['u'] = u1.reshape(2,-1).T
mesh.nodedata['p'] = p1
mesh.to_vtk(fname=fname)

BForm = solver.Ossen_BForm()
LForm = solver.Ossen_LForm()

BC = DirichletBC(space=(uspace,pspace), 
        gd=(pde.velocity, pde.pressure), 
        threshold=(None, None), 
        method='interp')
'''
BC = DirichletBC(space=(uspace,pspace), 
        gd=(pde.velocity, pde.pressure), 
        threshold=(pde.is_u_boundary, pde.is_p_boundary), 
        method='interp')
'''

for i in range(10):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.NS_update(u1)
    A = BForm.assembly()
    b = LForm.assembly()
    A,b = BC.apply(A,b)

    x = spsolve(A, b, 'mumps')
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]
    
    print(mesh.error(u1, pde.velocity))
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    mesh.to_vtk(fname=fname)
    
    timeline.advance()
