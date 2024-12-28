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

solver = NSFEMSolver(pde, mesh, pspace, uspace, dt, q=5)

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

u0 = uspace.function()
us = uspace.function()
u1 = uspace.function()
p0 = pspace.function()
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['u'] = u1.reshape(2,-1).T
mesh.nodedata['p'] = p1
mesh.to_vtk(fname=fname)

BC = DirichletBC(space=uspace, 
        gd=pde.velocity, 
        threshold=pde.is_u_boundary, 
        method='interp')

BForm0 = solver.IPCS_BForm_0(None)
LForm0 = solver.IPCS_LForm_0()
A0 = BForm0.assembly()   

for i in range(1):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.update_ipcs_0(u0, p0)
    print(bm.sum(bm.abs(A0.to_dense())))
    b0 = LForm0.assembly()
    A0,b0 = BC.apply(A0,b0)
    print(bm.sum(bm.abs(b0)))

    us[:] = spsolve(A0, b0, 'mumps')
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    mesh.to_vtk(fname=fname)
    
    timeline.advance()
