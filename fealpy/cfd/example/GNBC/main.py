#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: main.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 04:00:47 PM CST
	@bref 
	@ref 
'''  

# import os
# os.system(f'rm *.vtu')

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.solver import spsolve, cg, gmres 
from fealpy.fem import DirichletBC

from pde import CouetteFlow
from solver import Solver

from fealpy.utils import timer

bm.set_backend('pytorch')
#bm.set_backend('numpy')
#bm.set_default_device('cuda')

output = './'
h = 1/256
T = 2
nt = int(T/(0.1*h))

pde = CouetteFlow(h=h)
mesh = pde.mesh()
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt
time = timer()
next(time)
phispace = LagrangeFESpace(mesh, p=1)
pspace = LagrangeFESpace(mesh, p=0, ctype='D')
# pspace = LagrangeFESpace(mesh, p=1)
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

# fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi'] = phi0
mesh.nodedata['u'] = u0.reshape(2,-1).T
#mesh.celldata['p'] = p1
mesh.nodedata['mu'] = mu1
# mesh.to_vtk(fname=fname)

CH_BForm = solver.CH_BForm()
CH_LForm = solver.CH_LForm()
NS_BForm = solver.NS_BForm()
NS_LForm = solver.NS_LForm()

is_up = space.is_boundary_dof(pde.is_up_boundary)
is_down = space.is_boundary_dof(pde.is_down_boundary)
#NS_BC = DirichletBC(space=(uspace,pspace), \
#        gd=(pde.u_w, pde.p_dirichlet), \
#        threshold=(pde.is_wall_boundary, pde.is_p_dirichlet), method='interp')

is_uy_bd = space.is_boundary_dof(pde.is_uy_Dirichlet)
ux_gdof = space.number_of_global_dofs()
is_bd = bm.concatenate((bm.zeros(ux_gdof, dtype=bool), is_uy_bd, bm.zeros(pgdof, dtype=bool)))
NS_BC = DirichletBC(space=(uspace,pspace), \
        gd=bm.zeros(ugdof+pgdof, dtype=bm.float64), \
        threshold=is_bd, method='interp')

time.send("初始化用时")
for i in range(nt):
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

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['u'] = u2.reshape(2,-1).T
    #mesh.celldata['p'] = p2
    mesh.nodedata['mu'] = mu2
    # mesh.to_vtk(fname=fname)
    timeline.advance()
    time.send(f"第{i+1}次画图用时")
    uuu = u2.reshape(2,-1).T
    print("上边界最大值",bm.max(uuu[is_up,0]))
    print("上边界最小值",bm.min(uuu[is_up,0]))
    print("下边界最大值",bm.max(uuu[is_down,0]))
    print("下边界最小值",bm.min(uuu[is_down,0]))
#next(time)

