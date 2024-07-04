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
from fealpy.utils import timer
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.navier_stokes_equation_2d import Poisuille

from fealpy.torch.typing import Tensor, Index
from fealpy.torch.mesh import TriangleMesh

from fealpy.torch.functionspace import LagrangeFESpace
from fealpy.torch.functionspace import TensorFunctionSpace

from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarConvectionIntegrator,
    ScalarMassIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)

from fealpy.decorator import barycentric

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ns = 16
output = './'
udegree = 2
pdegree = 1
T = 10
nt = 1000
q = 4
pde = Poisuille()
rho = pde.rho
mu = pde.mu


time = timer()
next(time)
mesh = TriangleMesh.from_box(pde.domain(), nx = ns, ny = ns)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=pdegree)
uspace = TensorFunctionSpace(LagrangeFESpace(mesh, p=udegree), (2, ), dof_last=False)
time.send("mesh_and_space")

u0 = uspace.function(dim=2)
u1 = uspace.function(dim=2)
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u0 
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)
'''
uspace.doforder='vdims'
kwargs = {'dtype': torch.float64, "device": device}
bcs = torch.tensor([[1,0,0],[0,1,0]], **kwargs)
print(u0(bcs))
'''
uspace.doforder = 'vdims'
@barycentric
def coefu0(bcs, index=_S):
    kwargs = {'dtype': bcs.dtype, "device": device}
    return u0(bcs, index).to(**kwargs)

bform = BilinearForm(uspace)
bform.add_integrator(ScalarDiffusionIntegrator(rho, q=q))
bform.add_integrator(ScalarMassIntegrator(coef=mu, q=q))
MS = bform.assembly()
time.send('forms')




#for i in range(0,nt):












'''
@barycentric
def ocoef(bcs, index):
    return ou0(bcs,index)

obform = oBilinearForm(ouspace)
obform.add_domain_integrator(oSC(c=ocoef, q=4))
oC = obform.assembly() 

_S = slice(None)


@barycentric
def coef(bcs:Tensor, index=_S):
    kwargs = {'dtype': bcs.dtype, "device": bcs.device}
    return u0(bcs, index).to(**kwargs)
integrator = ScalarConvectionIntegrator(coef, q=4)
bcs, ws, phi, gphi, cm, index = integrator.fetch(uspace)

a = coef(bcs)
bform = BilinearForm(uspace)
bform.add_integrator(ScalarConvectionIntegrator(coef, q=4))
C = bform.assembly() 
'''





