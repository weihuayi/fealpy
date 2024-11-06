#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: main.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 04:00:47 PM CST
	@bref 
	@ref 
'''  

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.solver import spsolve, cg, gmres 

from pde import CouetteFlow
from solver import Solver

output = './'
h = 1/256
#h = 1/128
T = 2
nt = int(T/(0.1*h))

pde = CouetteFlow()
mesh = pde.mesh(h)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

phispace = LagrangeFESpace(mesh, p=1)
pspace = LagrangeFESpace(mesh, p=0, ctype='D')
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))

solver = Solver(pde, mesh, pspace, phispace, uspace, dt)

u0 = uspace.function()
u1 = uspace.function()
u2 = uspace.function()
phi0 = phispace.interpolate(pde.init_phi)
phi1 = phispace.function()
# TODO:第一步求解
phi1[:] = phi0[:]
phi2 = phispace.function()
mu1 = phispace.function()
mu2 = phispace.function()
p1 = pspace.function()
p2 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
phigdof = phispace.number_of_global_dofs()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi'] = phi0
mesh.nodedata['u'] = u0
mesh.nodedata['p'] = p1
mesh.nodedata['mu'] = mu1
mesh.to_vtk(fname=fname)

CH_BForm = solver.CH_BForm()
CH_LForm = solver.CH_LForm()
NS_BForm = solver.NS_BForm()
NS_LForm = solver.NS_LForm()

exit()
for i in range(100):
    t = timeline.next_time_level()
    print("time=", t)

    solver.CH_update(u0, u1, phi0, phi1)
    CH_A = CH_BForm.assembly()
    CH_b = CH_LForm.assembly()
    CH_x = spsolve(CH_A, CH_b, 'mumps')
    
    phi2[:] = CH_x[:phigdof]
    mu2[:] = CH_x[phigdof:] 

    solver.NS_update(u0, u1, mu2, phi2)
    NS_A = NS_BForm.assembly()
    NS_b = NS_LForm.assembly()
    NS_x = spsolve(NS_A, NS_b, 'mumps') 
    u2[:] = NS_x[:ugdof]
    p1[:] = NS_x[ugdof:]
    
    u0[:] = u1[:]
    u1[:] = u2[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['u'] = u2
    mesh.nodedata['p'] = p2
    mesh.nodedata['mu'] = mu2
    mesh.to_vtk(fname=fname)
    timeline.advance()














'''
bcs = bm.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])
qf = mesh.quadrature_formula(q=3, etype='face')
bcs, ws = qf.get_quadrature_points_and_weights()
print(u0(bcs).shape)
print(u0.grad_value(bcs).shape)
print(phi2.grad_value(bcs).shape)
print(phi2(bcs).shape)
'''
