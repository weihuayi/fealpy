#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_cylinder_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 12 Aug 2024 04:52:25 PM CST
	@bref 
	@ref 
'''  
#from fealpy import logger
#logger.setLevel('ERROR')

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator, 
        ScalarMassIntegrator,  PressWorkIntegrator ,
        ScalarConvectionIntegrator)

from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.sparse import COOTensor
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import LinearBlockForm, BlockForm
from fealpy.solver import spsolve 

from fealpy.functionspace import Function 


from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder
from fealpy.decorator import barycentric, cartesian
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.fem import DirichletBC

#TODO:mesh.nodedata对tensorspace的情况
#TODO:boundary_interpolate的不同情况

backend = 'pytorch'
#backend = 'numpy'
device = 'cuda'
#device = 'cpu'
bm.set_backend(backend)
bm.set_default_device(device)

output = './'
udegree = 2
pdegree = 1
q = 4
T = 5
nt = 5000
pde = FlowPastCylinder()
rho = pde.rho
mu = pde.mu

mesh = pde.mesh(0.05, device = device)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

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

A_bform = BilinearForm(uspace)
A_bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
A_bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q)) 
ConvectionIntegrator = ScalarConvectionIntegrator(q=q)
A_bform.add_integrator(ConvectionIntegrator)

##LinearForm
ulform = LinearForm(uspace)
SourceIntegrator = ScalarSourceIntegrator(q = q)
ulform.add_integrator(SourceIntegrator)
plform = LinearForm(pspace)

## b
#u_dirichlet,is_u_boundary = uspace.boundary_interpolate(pde.u_dirichlet, threshold = pde.is_u_boundary, method='interp')
#p_dirichlet,is_p_boundary = pspace.boundary_interpolate(pde.p_dirichlet, threshold = pde.is_p_boundary, method='interp')

for i in range(10):
    t1 = timeline.next_time_level()
    print("time=", t1)

    ConvectionIntegrator.coef = u0
    ConvectionIntegrator.clear()
    A = BlockForm([[A_bform, P_bform],
                   [P_bform.T, None]])
    A = A.assembly()
    
    SourceIntegrator.source = u0
    SourceIntegrator.clear() 
    b = LinearBlockForm([ulform, plform]).assembly()
    
    A,b = DirichletBC((uspace,pspace), gD=(pde.u_dirichlet, pde.p_dirichlet), 
                      threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp').apply(A, b)

    #A,b = DirichletBC((uspace,pspace), (u_dirichlet, bm.zeros(pgdof)), 
    #                  threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp').apply(A, b)
    #A,b = DirichletBC((uspace,pspace), (u_dirichlet, pde.p_dirichlet), 
    #                  threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp').apply(A, b)
    #A,b = DirichletBC((uspace,pspace), bm.concatenate((u_dirichlet[:], bm.zeros(pgdof))), 
    #                  threshold=(pde.is_u_boundary, pde.is_p_boundary), method='interp').apply(A, b)
    #A,b = DirichletBC((uspace,pspace), gD=(pde.u_dirichlet, pde.p_dirichlet), 
    #                  threshold=(is_u_boundary, is_p_boundary), method='interp').apply(A, b)
    #A,b = DirichletBC((uspace,pspace), bm.concatenate((u_dirichlet[:], bm.zeros(pgdof))), 
    #                  threshold=bm.concatenate((is_u_boundary, is_p_boundary)), method='interp').apply(A, b)
    
    x = spsolve(A, b, 'mumps')
    
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    
    mesh.nodedata['velocity'] = u1.reshape(2,-1).T
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
        
    u0[:] = u1[:] 
    timeline.advance()
print(bm.sum(bm.abs(u1[:])))
