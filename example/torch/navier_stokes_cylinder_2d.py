#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_cylinder.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 27 Jun 2024 07:59:03 PM CST
	@bref 
	@ref 
'''  
import torch
import numpy as np
import scipy.sparse as sp

from fealpy.utils import timer
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder

from fealpy.torch.typing import Tensor, Index
from fealpy.torch.mesh import TriangleMesh
from fealpy.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh as OTM
from fealpy.torch.solver import sparse_cg

from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.functionspace import TensorFunctionSpace

from fealpy.decorator import barycentric, cartesian
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarConvectionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
    PressWorkIntegrator,PressWorkIntegrator1,
    DirichletBC
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

tmr = timer()
next(tmr)

output = './'
udegree = 2
pdegree = 1
T = 5
nt = 5000
q = 3
eps=1e-12
pde = FlowPastCylinder()
rho = pde.rho
mu = pde.mu
fwargs = {'dtype': torch.float64, "device": device}
iwargs = {'dtype': torch.int32, "device": device}

@cartesian
def is_inflow_boundary( p:Tensor):
    return torch.abs(p[..., 0]) < eps

@cartesian
def is_outflow_boundary(p:Tensor):
    return torch.abs(p[..., 0] - 2.2) < eps

@cartesian
def is_circle_boundary( p:Tensor):
    x = p[...,0]
    y = p[...,1]
    return (torch.sqrt(x**2 + y**2) - 0.05) < eps
  
@cartesian
def is_wall_boundary( p:Tensor):
    return (torch.abs(p[..., 1] -0.41) < eps) | \
           (torch.abs(p[..., 1] ) < eps)

@cartesian
def u_inflow_dirichlet( p:Tensor):
    x = p[...,0]
    y = p[...,1]
    value = torch.zeros(p.shape, **fwargs)
    value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
    value[...,1] = 0
    return value

omesh = pde.mesh1(0.01)
node = torch.from_numpy(omesh.entity('node')).to(**fwargs)
cell = torch.from_numpy(omesh.entity('cell')).to(**iwargs)
mesh = TriangleMesh(node, cell)

timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=pdegree)
uspace = LagrangeFESpace(mesh, p=udegree)

u0 = uspace.function(dim=2)
u1 = uspace.function(dim=2)
p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u0 
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)

uspace.doforder = 'vdims'

bform = BilinearForm(uspace)
bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
M = bform.assembly()

bform = BilinearForm(uspace)
bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q))
S = bform.assembly()

bform = BilinearForm((pspace, uspace))
bform.add_integrator(PressWorkIntegrator(q=q)) 
APX = bform.assembly()

bform = BilinearForm((pspace, uspace))
bform.add_integrator(PressWorkIntegrator1(q=q)) 
APY = bform.assembly()

#边界处理
xx = torch.zeros(gdof, **fwargs)
u_isbddof_u0 = uspace.is_boundary_dof()
u_isbddof_in = uspace.is_boundary_dof(threshold = is_inflow_boundary)
u_isbddof_out = uspace.is_boundary_dof(threshold = is_outflow_boundary)
u_isbddof_circle = uspace.is_boundary_dof(threshold = is_circle_boundary)

u_isbddof_u0[u_isbddof_in] = False 
u_isbddof_u0[u_isbddof_out] = False 
xx[0:ugdof][u_isbddof_u0] = 0
xx[ugdof:2*ugdof][u_isbddof_u0] = 0

u_isbddof = u_isbddof_u0
u_isbddof[u_isbddof_in] = True
ipoint = uspace.interpolation_points()[u_isbddof_in]
uinfow = u_inflow_dirichlet(ipoint)
xx[0:ugdof][u_isbddof_in] = uinfow[:,0]
xx[ugdof:2*ugdof][u_isbddof_in] = uinfow[:,1]

p_isBdDof_p0 = pspace.is_boundary_dof(threshold =  is_outflow_boundary) 
xx[2*ugdof:][p_isBdDof_p0] = 0 
isBdDof = torch.cat([u_isbddof, u_isbddof, p_isBdDof_p0], dim=0)


for i in range(nt):
    t1 = timeline.next_time_level()
    print("time=", t1)
    
    @barycentric
    def concoef(bcs:Tensor, index):
        kwargs = {'dtype': bcs.dtype, "device": bcs.device}
        return u0(bcs, index).to(**kwargs)
    
    bform = BilinearForm(uspace)
    bform.add_integrator(ScalarConvectionIntegrator(concoef, q=4))
    C = bform.assembly() 
    
    indices = torch.tensor([[],[]], **iwargs)
    data = torch.tensor([], **fwargs)
    zeros_0 = torch.sparse_coo_tensor(indices, data, (ugdof,ugdof))
    zeros_1 = torch.sparse_coo_tensor(indices, data, (pgdof,pgdof))
    
    A0 = torch.cat([M+S+C, zeros_0, -APX], dim=1)
    A1 = torch.cat([zeros_0, M+S+C, -APY], dim=1)
    A2 = torch.cat([-APX.T, -APY.T ,zeros_1], dim=1)
    A = torch.cat((A0,A1,A2),dim=0)
    
    b0 = M@u0[:,0] 
    b1 = M@u0[:,1]
    b2 = torch.zeros(pgdof, **fwargs) 
    b = torch.cat([b0,b1,b2])
    
    b -= A@xx
    b[isBdDof] = xx[isBdDof]
    
    A = A.coalesce()
    indices = A.indices()
    new_values = A.values().clone()
    IDX = isBdDof[indices[0, :]] | isBdDof[indices[1, :]]
    new_values[IDX] = 0
    A = torch.sparse_coo_tensor(indices, new_values, A.size(), **fwargs)
    index, = torch.nonzero(isBdDof, as_tuple=True) 
    one_values = torch.ones(len(index))
    one_indices = torch.stack([index,index], dim=0)
    A1 = torch.sparse_coo_tensor(one_indices, one_values, A.size(), **fwargs)
    A += A1 
    A = A.coalesce()
    
    values = A.values().cpu().numpy()
    indices = A.indices().cpu().numpy()
    A = sp.coo_matrix((values, (indices[0], indices[1])), shape=A.shape) 
    A = A.tocsr()
    b = b.cpu().numpy()
    
    x = sp.linalg.spsolve(A,b)
    x = torch.from_numpy(x).to(**fwargs)
    #x = sparse_cg(A, b, maxiter=10000)

    u1[:,0] = x[:ugdof]
    u1[:,1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['velocity'] = u1
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
    
    u0[:] = u1
    timeline.advance() 
