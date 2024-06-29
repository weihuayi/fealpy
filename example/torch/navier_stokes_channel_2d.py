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
from torch import Tensor
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace
import numpy as np
from fealpy.mesh import IntervalMesh
from fealpy.utils import timer
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarConvectionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.navier_stokes_equation_2d import Poisuille

from fealpy.mesh import TriangleMesh as OTM
from fealpy.fem import ScalarConvectionIntegrator as oSC
from fealpy.fem import BilinearForm as oBilinearForm
from fealpy.decorator import barycentric
from fealpy.functionspace import LagrangeFESpace as OLFE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pde = Poisuille()
mesh = TriangleMesh(node, cell)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

udim = 2
udegree = 2
pdegree = 1

uspace = LagrangeFESpace(mesh,p=udegree)
pspace = LagrangeFESpace(mesh,p=pdegree)
tmr.send("mesh_and_space")

ouspace = OLFE(omesh,p=udegree)
ou0 = ouspace.function(dim=udim)

u0 = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)
p1 = pspace.function()


output = './'
fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u0 
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)



for i in range(0,nt):












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





