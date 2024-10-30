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

from pde import CouetteFlow
from solver import Solver

output = './'
h = 1/256
dt = 0.1*h
pde = CouetteFlow()
mesh = pde.mesh(h)

phispace = LagrangeFESpace(mesh, p=1)
pspace = LagrangeFESpace(mesh, p=0)
space = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(space, (2,-1))
solver = Solver(pde, mesh, pspace, phispace, uspace, dt)

u0 = uspace.function()
u1 = uspace.function()
u2 = uspace.function()
phi0 = phispace.interpolate(pde.init_phi)
phi1 = phispace.function()
phi2 = phispace.function()
p0 = pspace.function()
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi'] = phi0
mesh.to_vtk(fname=fname)

CH_BForm = solver.CH_BForm()
CH_LForm = solver.CH_LForm()

NS_BForm = solver.CH_BForm()
NS_LForm = solver.CH_LForm()

