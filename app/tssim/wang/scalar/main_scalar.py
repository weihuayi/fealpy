#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: main.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 04:00:47 PM CST
	@bref 
	@ref 
'''  
import os


from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.solver import spsolve, cg, gmres 
from fealpy.fem import DirichletBC

from pde import CouetteFlow
from solver import Solver

output = './'
h = 1/256
#h = 1/16
T = 2
nt = int(T/(0.1*h))

pde = CouetteFlow()
mesh = pde.mesh(h)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

phispace = LagrangeFESpace(mesh, p=1)
#pspace = LagrangeFESpace(mesh, p=0, ctype='D')
pspace = LagrangeFESpace(mesh, p=1)
uspace = LagrangeFESpace(mesh, p=2)

solver = Solver(pde, mesh, pspace, phispace, uspace, dt, q=5)

u0x = uspace.function()
u1x = uspace.function()
u2x = uspace.function()
u0y = uspace.function()
u1y = uspace.function()
u2y = uspace.function()
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
mesh.nodedata['ux'] = u0x
mesh.nodedata['uy'] = u0y
#mesh.celldata['p'] = p1
mesh.nodedata['mu'] = mu1
mesh.to_vtk(fname=fname)

CH_BForm = solver.CH_BForm()
CH_LForm = solver.CH_LForm()
NS_BForm = solver.NS_BForm()
NS_LForm = solver.NS_LForm()

is_uy_bd = uspace.is_boundary_dof(pde.is_uy_Dirichlet)
is_bd = bm.concatenate((bm.zeros(ugdof, dtype=bool), is_uy_bd, bm.zeros(pgdof, dtype=bool)))
NS_BC = DirichletBC(space=(uspace,uspace,pspace), \
        gd=bm.zeros(2*ugdof+pgdof), \
        threshold=is_bd, method='interp')

for i in range(3):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.CH_update(u0x, u0y, u1x, u1y, phi0, phi1)
    CH_A = CH_BForm.assembly()
    CH_b = CH_LForm.assembly()
    CH_x = spsolve(CH_A, CH_b, 'mumps')
    
    phi2[:] = CH_x[:phigdof]
    mu2[:] = CH_x[phigdof:] 

    solver.NS_update(u0x, u0y, u1x, u1y, mu2, phi2, phi1)
    NS_A = NS_BForm.assembly()
    NS_b = NS_LForm.assembly()
    NS_A,NS_b = NS_BC.apply(NS_A,NS_b)
     
    NS_x = spsolve(NS_A, NS_b, 'mumps') 
    u2x[:] = NS_x[:ugdof]
    u2y[:] = NS_x[ugdof:-pgdof]
    p2[:] = NS_x[-pgdof:]
    
    u0x[:] = u1x[:]
    u1x[:] = u2x[:]
    u0y[:] = u1y[:]
    u1y[:] = u2y[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['ux'] = u2x
    mesh.nodedata['uy'] = u2y
    #mesh.celldata['p'] = p2
    mesh.nodedata['mu'] = mu2
    mesh.to_vtk(fname=fname)
    timeline.advance()


