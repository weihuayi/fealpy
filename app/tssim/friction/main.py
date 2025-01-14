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
#os.system(f'rm *.vtu')

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace 
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.solver import spsolve, cg, gmres 
from fealpy.fem import DirichletBC
from pde import slip_stick
from solver import Solver

from fealpy.utils import timer

#bm.set_backend('pytorch')
bm.set_backend('numpy')
#bm.set_default_device('cuda')

output = './'
h = 1/250
#h = 1/100
T = 1
nt = int(T/(0.1*h))
hh = 3*h

pde = slip_stick(h=h)
mesh = pde.mesh()


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

phispace = LagrangeFESpace(mesh, p=1)
#pspace = LagrangeFESpace(mesh, p=0, ctype='D')
pspace = LagrangeFESpace(mesh, p=1)
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))

solver = Solver(pde, mesh, pspace, phispace, uspace, dt, q=5)
u0 = uspace.function()
u1 = uspace.function()
u2 = uspace.function()
phi0 = phispace.interpolate(pde.init_interface)
phi1 = phispace.function()

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


bdface_index = mesh.boundary_face_index()
bdface_bc = mesh.entity_barycenter('face', index=bdface_index)
stick_index = bdface_index[pde.is_wall_boundary(bdface_bc)]
new_slip_set = set()
allslip_set = set()
e2dof = space.edge_to_dof()
slip_threshold = 1.05

control = True
for i in range(nt):
    t = timeline.next_time_level()
    print(f"第{i+1}步")
    print("time=", t)

    solver.CH_update(u0, u1, phi0, phi1)
    CH_A = CH_BForm.assembly()
    CH_b = CH_LForm.assembly()
    CH_x = spsolve(CH_A, CH_b, 'mumps')
    
    phi2[:] = CH_x[:phigdof]
    mu2[:] = CH_x[phigdof:] 

    solver.NS_update(u0, u1, mu2, phi2, phi1)
    NS_A = NS_BForm.assembly()
    NS_b = NS_LForm.assembly()
    NS_A,NS_b = NS_BC.apply(NS_A,NS_b)
    NS_x = spsolve(NS_A, NS_b, 'mumps') 
    u2[:] = NS_x[:ugdof]
    p2[:] = NS_x[ugdof:]
    
    u0[:] = u1[:]
    u1[:] = u2[:]
    phi0[:] = phi1[:]
    phi1[:] = phi2[:]
    mu1[:] = mu2[:]
    p1[:] = p2[:]
    
    stress = solver.stress(u1)
    
    up_node, down_node = solver.interface_on_boundary(phi1)
    up_dof, down_dof = solver.slip_dof(up_node, down_node, hh)
    value = bm.mean(bm.abs(stress[up_dof,0,1]))
    print("值:",value)
    if (i>50) & (value > 30) & control:
        print("开始slip")
        from fealpy.decorator import barycentric,cartesian
        @cartesian
        def is_slip_index(p):
            x = p[..., 0]
            y = p[..., 1]
            tag_up= bm.abs(y-0.15)<1e-10
            tag1 = (x > up_node[0]) & (x < up_node[0] + hh)  
            tag_down= bm.abs(y-0)<1e-10
            tag2 = (x > down_node[0]) & (x < down_node[0] + hh)
            tag_up = tag_up & tag1
            tag_down = tag_down & tag2
            return tag_up | tag_down
        
        
        slip_index = bdface_index[is_slip_index(bdface_bc)]
        new_slip_set = new_slip_set | set(slip_index) 
        allslip_set = allslip_set | set(slip_index)


        stick_index = bm.array(list(set(stick_index) - set(slip_index)), dtype=bm.int32)

        solver.is_bd = stick_index
        CH_BForm = solver.CH_BForm()
        CH_LForm = solver.CH_LForm()
        NS_BForm = solver.NS_BForm()
        NS_LForm = solver.NS_LForm()
    
    new_slip_index = bm.array(list(new_slip_set), dtype=bm.int32)
    slip_value = bm.mean(bm.abs(stress[e2dof[new_slip_index],0,1]))
    print("滑移的值",slip_value)
    if slip_value < 1.42:
        control = False
        print("control is False")
        if slip_value < slip_threshold:
            control = True
            new_slip_set = set()
            print("control is True")


        
        
    #solver.plot_change_on_y(bm.abs(stress[:,0,1]), y=0.15, space=space)
    #cc = bm.argmax(stress[:,0,1])
    #ip = uspace.interpolation_points()

    print("上边界",up_node)
    print("下边界",down_node)
    #print(ip[bm.where(up_dof)])
    #print(ip[bm.where(down_dof)])

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi2
    mesh.nodedata['u'] = u2.reshape(2,-1).T
    #mesh.celldata['p'] = p2
    mesh.nodedata['p'] = p2
    mesh.nodedata['mu'] = mu2
    mesh.nodedata['stress0'] = stress[:,0,1]

    mesh.to_vtk(fname=fname)
    timeline.advance()
    uuu = u2.reshape(2,-1).T
#next(time)
