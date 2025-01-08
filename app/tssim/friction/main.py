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
os.system(f'rm *.vtu')

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.solver import spsolve, cg, gmres 
from fealpy.fem import DirichletBC
import matplotlib.pyplot as plt
from pde import slip_stick
from solver import Solver

from fealpy.utils import timer

#bm.set_backend('pytorch')
bm.set_backend('numpy')
#bm.set_default_device('cuda')

output = './'
#h = 1/250
h = 1/100
T = 1
nt = int(T/(0.1*h))

pde = slip_stick(h=h)
mesh = pde.mesh()


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt
time = timer()
next(time)

phispace = LagrangeFESpace(mesh, p=1)
#pspace = LagrangeFESpace(mesh, p=0, ctype='D')
pspace = LagrangeFESpace(mesh, p=1)
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))

'''
ipoint = space.interpolation_points()
import matplotlib.pylab  as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#mesh.find_edge(axes,fontsize=20,showindex=True)
mesh.find_node(axes,node=ipoint,fontsize=20,showindex=True)
plt.show()
'''

solver = Solver(pde, mesh, pspace, phispace, uspace, dt, q=5)
u0 = uspace.function()
u1 = uspace.function()
u2 = uspace.function()
phi0 = phispace.interpolate(pde.init_interface)
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
sugdof = space.number_of_global_dofs()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi'] = phi0
mesh.nodedata['u'] = u0.reshape(2,-1).T
#mesh.celldata['p'] = p1
mesh.nodedata['p'] = p1
mesh.nodedata['mu'] = mu1
mesh.to_vtk(fname=fname)

CH_BForm = solver.CH_BForm()
CH_LForm = solver.CH_LForm()
NS_BForm = solver.NS_BForm()
NS_LForm = solver.NS_LForm()

is_bd = uspace.is_boundary_dof((pde.is_ux_Dirichlet, pde.is_uy_Dirichlet), method='interp')
is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))

uin_gd,_ = uspace.boundary_interpolate(pde.u_inflow_dirichlet, threshold=pde.is_left_boundary, method='interp')
gd = bm.concatenate((uin_gd[:], bm.zeros(pgdof, dtype=bm.float64)))

NS_BC = DirichletBC(space=(uspace,pspace), \
        gd=gd, \
        threshold=is_bd, method='interp')

time.send("初始化用时")

for i in range(2):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.CH_update(u0, u1, phi0, phi1)
    CH_A = CH_BForm.assembly()
    CH_b = CH_LForm.assembly()
    time.send(f"第{i+1}次CH组装用时")
    CH_x = spsolve(CH_A, CH_b, 'mumps')
    time.send(f"第{i+1}次CH求解用时")
    
    phi2[:] = CH_x[:phigdof]
    mu2[:] = CH_x[phigdof:] 

    solver.NS_update(u0, u1, mu2, phi2, phi1)
    NS_A = NS_BForm.assembly()
    NS_b = NS_LForm.assembly()
    NS_A,NS_b = NS_BC.apply(NS_A,NS_b)
    time.send(f"第{i+1}次NS组装用时") 
    NS_x = spsolve(NS_A, NS_b, 'mumps') 
    time.send(f"第{i+1}次NS求解用时")
    u2[:] = NS_x[:ugdof]
    p2[:] = NS_x[ugdof:]
    
    u0[:] = u1[:]
    u1[:] = u2[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]
    
    stress = uspace.grad_recovery(u1)
    stress = 0.5*(stress + stress.swapaxes(-1,-2))
    normal = mesh.edge_unit_normal()
    tangent = mesh.edge_unit_tangent()
    e2dof = uspace.edge_to_dof()
    
    iswalldof = uspace.scalar_space.is_boundary_dof(pde.is_up_boundary, method='interp')
    u_tau = bm.abs(stress[iswalldof, 0,1])
    print(u_tau)

    inteface_phi = bm.where(bm.abs(phi1[:])<0.9)
    interface_node = mesh.node[inteface_phi]
    xx = bm.mean(interface_node[:,0])
    #print(xx)
    
    ip = uspace.interpolation_points()[iswalldof]
    tag = (ip[:,0]>xx) & (ip[:,0]<xx+5*h)
    #print(ip[tag])
    print(ip[:,0])
    print(bm.sort(ip[:,0]))
    print(bm.argsort(ip[:,0]))
    plt.plot(bm.sort(ip[:,0]), u_tau[bm.argsort(ip[:,0])])
    plt.show()
    



    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['u'] = u2.reshape(2,-1).T
    #mesh.celldata['p'] = p2
    mesh.nodedata['p'] = p2
    mesh.nodedata['mu'] = mu2
    mesh.nodedata['stress0'] = stress[:,0,1]



    mesh.to_vtk(fname=fname)
    timeline.advance()
    time.send(f"第{i+1}次画图用时")
    uuu = u2.reshape(2,-1).T
#next(time)
