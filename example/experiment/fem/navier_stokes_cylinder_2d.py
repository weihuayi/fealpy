#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_cylinder_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 12 Aug 2024 04:52:25 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator, 
        ScalarMassIntegrator,  PressWorkIntegrator ,
        ScalarConvectionIntegrator)

from fealpy.fem import LinearForm, SourceIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.fem import LinearBlockForm, BlockForm
from fealpy.solver import spsolve, cg 

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.fem import DirichletBC
from fealpy.utils import timer

#TODO:mesh.nodedata对tensorspace的情况

#backend = 'pytorch'
backend = 'numpy'
#device = 'cuda'
device = 'cpu'
bm.set_backend(backend)
#bm.set_default_device(device)

output = './numpy/'
udegree = 2
pdegree = 1
q = 4
T = 7
nt = 7000
pde = FlowPastCylinder()
rho = pde.rho
mu = pde.mu

mesh = pde.mesh(0.007, device = device)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt
#tmr = timer()
#next(tmr)

pspace = LagrangeFESpace(mesh, p=pdegree)
space = LagrangeFESpace(mesh, p=udegree)
uspace = TensorFunctionSpace(space, (2, -1))

u0 = uspace.function() 
u1 = uspace.function()
p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = ugdof + pgdof

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u1.reshape(2,-1).T
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)

##BilinearForm
P_bform = BilinearForm((pspace, uspace))
P_bform.add_integrator(PressWorkIntegrator(-1, q=q))

U_bform = BilinearForm(uspace)
M = ScalarMassIntegrator(rho/dt, q=q)
S = ScalarDiffusionIntegrator(mu, q=q)
D = ScalarConvectionIntegrator(q=q)
U_bform.add_integrator([M,S,D])
BForm = BlockForm([[U_bform, P_bform],
               [P_bform.T, None]])

##LinearForm
ulform = LinearForm(uspace)
f = SourceIntegrator(q = q)
ulform.add_integrator(f)
plform = LinearForm(pspace)
LBForm = LinearBlockForm([ulform, plform])

#边界处理
BC = DirichletBC((uspace,pspace), gd=(pde.u_dirichlet, pde.p_dirichlet), 
                      threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp')
#tmr.send('网格和pde生成时间')
print(f"总共自由度为:{gdof}")

for i in range(nt):
    t1 = timeline.next_time_level()
    print("time=", t1)

    #tmr.send("其他") 
    D.coef = rho*u0
    D.clear()
    A = BForm.assembly()
     
    f.source = u0/dt
    f.clear()
    b = LBForm.assembly()
    
    A,b = BC.apply(A,b)
    #tmr.send("边界处理") 
    
    x = spsolve(A, b, 'scipy')
    #x = cg(A, b)
    #tmr.send("求解") 
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    
    u0[:] = u1[:] 
    mesh.nodedata['velocity'] = u1.reshape(2,-1).T
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
        
    timeline.advance()
#next(tmr)
