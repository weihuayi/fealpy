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
        ScalarMassIntegrator, PressWorkIntegrator0, PressWorkIntegrator ,
        PressWorkIntegrator1, ScalarConvectionIntegrator)

from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.sparse import COOTensor
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BlockForm
from fealpy.solver import spsolve 

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder
from fealpy.decorator import barycentric, cartesian
from fealpy.old.timeintegratoralg import UniformTimeLine
from fealpy.fem import DirichletBC

#backend = 'pytorch'
backend = 'numpy'
bm.set_backend(backend)

output = './'
udegree = 2
pdegree = 1
q = 4
T = 5
nt = 5000
pde = FlowPastCylinder()
rho = pde.rho
mu = pde.mu

mesh = pde.mesh1(0.05)##改一下接口

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

## BilinearForm
P_bform = BilinearForm((pspace, uspace))
P_bform.add_integrator(PressWorkIntegrator(-1, q=q))

A_bform = BilinearForm(uspace)
A_bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
A_bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q)) 
ConvectionIntegrator = ScalarConvectionIntegrator(q=q)
A_bform.add_integrator(ConvectionIntegrator)
##LinearForm
lform = LinearForm(uspace)
SourceIntegrator = ScalarSourceIntegrator(q = q)
lform.add_integrator(SourceIntegrator)


#边界处理
## 边界处理太繁琐
xx = bm.zeros(gdof, dtype=mesh.ftype)

u_isbddof_u0 = space.is_boundary_dof()
u_isbddof_in = space.is_boundary_dof(threshold = pde.is_inflow_boundary)
u_isbddof_out = space.is_boundary_dof(threshold = pde.is_outflow_boundary)

u_isbddof_u0[u_isbddof_in] = False 
u_isbddof_u0[u_isbddof_out] = False 

u_isbddof = u_isbddof_u0
u_isbddof[u_isbddof_in] = True
ipoint = space.interpolation_points()
uinflow = pde.u_inflow_dirichlet(ipoint)

p_isBdDof_p0 = pspace.is_boundary_dof(threshold = pde.is_outflow_boundary) 
bd = bm.concatenate((u_isbddof_in, u_isbddof_in, p_isBdDof_p0))
value_bd = bm.concatenate((uinflow[:,0],uinflow[:,1], bm.zeros(pgdof)))
xx[bd] = value_bd[bd] 

isBdDof = bm.concatenate([u_isbddof, u_isbddof, p_isBdDof_p0], axis=0)


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
    b0 = lform.assembly()
    b2 = bm.zeros(pgdof) 
    b = bm.concatenate([b0, b2])
    
    ##LinearBlockForm
    b -= A@xx
    b[isBdDof] = xx[isBdDof]
    
    #A = DirichletBC(uspace, xx, threshold=isBdDof).apply_matrix(A, check=False)
    #A,b = DirichletBC(uspace, xx, isDDof=isBdDof).apply(A, b, check=None)
    diri = DirichletBC((uspace,pspace), xx ,threshold=isBdDof)
    A = diri.apply_matrix(A)
    b = diri.apply_vector(b, A)
    exit() 
    x = spsolve(A, b, 'mumps')
    
    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]
    

    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    ##mesh.nodedata 
    mesh.nodedata['velocity'] = u1.reshape(2,-1).T
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
    
    u0[:] = u1[:] 
    timeline.advance() 