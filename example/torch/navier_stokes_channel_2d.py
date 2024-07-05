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
import scipy.sparse as sp

import numpy as np
from fealpy.utils import timer
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.navier_stokes_equation_2d import Poisuille

from fealpy.torch.typing import Tensor, Index
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.solver import sparse_cg

from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.functionspace import TensorFunctionSpace
from fealpy.solver import  CupySolver


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

ns = 32
output = './'
udegree = 2
pdegree = 1
T = 10
nt = 1000
q = 4
pde = Poisuille()
rho = pde.rho
mu = pde.mu
fwargs = {'dtype': torch.float64, "device": device}
iwargs = {'dtype': torch.int32, "device": device}

@cartesian
def velocity(p:Tensor):
    x = p[...,0]
    y = p[...,1]
    value = torch.zeros(p.shape, **fwargs)
    value[...,0] = 4*y*(1-y)
    value[...,1] = 0 
    return value

@cartesian
def pressure(p:Tensor):
    x = p[..., 0]
    y = p[..., 1]
    val = 8*(1-x) 
    return val.to(**fwargs)

time = timer()
next(time)
mesh = TriangleMesh.from_box(pde.domain(), nx = ns, ny = ns, device=device)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=pdegree)
uspace = LagrangeFESpace(mesh, p=udegree)

#uspace = TensorFunctionSpace(LagrangeFESpace(mesh, p=udegree), (-1, 2))
time.send("mesh_and_space")

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
uso = uspace.function(dim=2)
pso = pspace.function()
uspace.interpolate(velocity, uso, dim=-2)
pspace.interpolate(pressure, pso)

xx = torch.zeros(gdof, **fwargs)
is_u_bdof = uspace.is_boundary_dof()
is_p_bdof = pspace.is_boundary_dof()
isBdDof = torch.cat([is_u_bdof,is_u_bdof,is_p_bdof])

xx[:ugdof][is_u_bdof] = uso[:,0][is_u_bdof]
xx[ugdof:2*ugdof][is_u_bdof] = uso[:,1][is_u_bdof]
xx[2*ugdof:][is_p_bdof] = pso[is_p_bdof]
errorMatrix = torch.zeros((4,nt), **fwargs)

solver  = CupySolver()
time.send('forms')


for i in range(5):
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
    
    time.send('ready')
    values = A.values().cpu().numpy()
    indices = A.indices().cpu().numpy()
    A = sp.coo_matrix((values, (indices[0], indices[1])), shape=A.shape) 
    A = A.tocsr()
    b = b.cpu().numpy()
    x = solver.cg_solver(A, b)
    x = torch.from_numpy(x).to(**fwargs)
    '''
    x = sparse_cg(A, b, maxiter=5000)
    ''' 
    
    time.send('solver')

    u1[:,0] = x[:ugdof]
    u1[:,1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    u0[:] = u1
    
    errorMatrix[2,i] = torch.max(torch.abs(uso-u1))
    errorMatrix[3,i] = torch.max(torch.abs(pso-p1))
    timeline.advance()
    
next(time)
