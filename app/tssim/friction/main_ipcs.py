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
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.cfd import NSFEMSolver
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve 
from pde import ChannelFlow
#from fealpy.pde.navier_stokes_equation_2d import ChannelFlow

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

BCu = DirichletBC(space=uspace, 
        gd=pde.velocity, 
        threshold=pde.is_u_boundary, 
        method='interp')

BCp = DirichletBC(space=pspace, 
        gd=pde.pressure, 
        threshold=pde.is_p_boundary, 
        method='interp')

BForm0 = solver.IPCS_BForm_0(threshold = None)
LForm0 = solver.IPCS_LForm_0()
AA0 = BForm0.assembly()   

Bform1 = solver.IPCS_BForm_1()
Lform1 = solver.IPCS_LForm_1()
AA1 = Bform1.assembly()

Bform2 = solver.IPCS_BForm_2()
Lform2 = solver.IPCS_LForm_2()
AA2 = Bform2.assembly()

print(bm.sum(bm.abs(AA2.toarray())))
for i in range(3):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)
     
    solver.update_ipcs_0(u0, p0)
    b0 = LForm0.assembly()
    A0,b0 = BCu.apply(AA0,b0)
    us[:] = spsolve(A0, b0, 'mumps')

    solver.update_ipcs_1(us, p0)
    b1 = Lform1.assembly()
    A1,b1 = BCp.apply(AA1, b1)
    p1[:] = spsolve(A1, b1, 'mumps')

    solver.update_ipcs_2(us, p0, p1)
    b2 = Lform2.assembly()
    u1[:] = spsolve(AA2, b2, 'mumps')
    pp = pspace.grad_recovery(p1, method='simple')
    bcs = bm.array([[1/3,1/3,1/3]])
    uu = uspace.grad_recovery(u1)
    ug = u1.grad_value(bcs)


    u0[:] = u1
    p0[:] = p1
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['u'] = u1.reshape(2,-1).T
    mesh.nodedata['p'] = p1
    mesh.nodedata['pp'] = pp.reshape(2,-1).T
    mesh.celldata['pg'] = p1.grad_value(bcs)[:,0,:]
    mesh.nodedata['uu'] = uu[:,0,0]
    mesh.celldata['ug'] = ug[:,:,0,0]
    mesh.to_vtk(fname=fname)
    #print(mesh.error(pde.velocity, u1))

    timeline.advance()
