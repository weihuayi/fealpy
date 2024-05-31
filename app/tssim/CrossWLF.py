#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 18 May 2024 03:01:14 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from CrossWLFSimulator import CrossSolver
from CrossWLFModel import CrossWLF
from fealpy.mesh import TriangleMesh
from fealpy.cfd import NSFEMSolver
from fealpy.levelset.ls_fem_solver import LSFEMSolver
from fealpy.functionspace import LagrangeFESpace
from fealpy.timeintegratoralg import UniformTimeLine
import matplotlib.pyplot as plt
from fealpy.decorator import barycentric, cartesian

ns = 10
T = 10
nt = 2000

pde = CrossWLF()
mesh = TriangleMesh.from_box(pde.box, 10*ns, ns)
uspace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
pspace = LagrangeFESpace(mesh, p=1, doforder='sdofs')
Tspace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
phispace = LagrangeFESpace(mesh, p=2, doforder='sdofs')
solver  = CrossSolver(pde, mesh, uspace,pspace, phispace)
timeline = UniformTimeLine(0,T,nt)
dt = timeline.dt
LSSolver = LSFEMSolver(phispace)
NSSolver = NSFEMSolver(mesh, dt ,uspace, pspace) 
inletdof = uspace.is_boundary_dof(pde.is_inlet_boundary)
walldof = uspace.is_boundary_dof(pde.is_wall_boundary)
phi0 = phispace.interpolate(pde.init_surface)
phi1 = phispace.function()
u0 = uspace.function(dim=2)
u0[0,inletdof] = 5
p0 = pspace.function()
T0 = Tspace.function()
T0[walldof] = 525

for i in range(1):
    phi1[:] = LSSolver.mumps_solve(5, phi0, dt, u0)
    
    #eta_l = solver.eta_l(T0, p0, u0) 
    
    rho = solver.parfunction(pde.rho, phi1)
    c = solver.parfunction(pde.c, phi1)
    lam = solver.parfunction(pde.lam, phi1)
    #eta = solver.parfunction(eta_l/pde.eta_g, phi1)
    #netwon流情况
    eta = solver.parfunction(0.001, phi1)
    
    #Re = solver.Re(eta_l)
    Re = 0.01
    #Br = solver.Br(eta_l)
    Br = 147398.844

    delta = solver.delta_epsion(phi1)
    kappa = solver.kappa(phi1)
    
    #ipcs
    A0 = NSSolver.ipcs_A_0(mu=eta, rho=rho, threshold=pde.is_outflow_boundary)
    b0 = NSSolver.ipcs_b_0(un=u0, p0=p0, source=1, rho=rho, threshold=pde.is_outflow_boundary)

'''
fname = 'test' + 'test_.vtu'
mesh.nodedata['u0'] = u0.transpose(1,0)
mesh.nodedata['phi1'] = phi1
mesh.nodedata['phi0'] = phi0
mesh.nodedata['rho'] = rho
mesh.to_vtk(fname=fname)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=mesh.interpolation_points(p=2)[walldof],color='r')
#plt.show()
'''
