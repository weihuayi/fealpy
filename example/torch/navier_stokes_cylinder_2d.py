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
from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder as PDE

from fealpy.mesh import TriangleMesh as OTM
from fealpy.fem import ScalarConvectionIntegrator as oSC
from fealpy.fem import BilinearForm as oBilinearForm
from fealpy.decorator import barycentric
from fealpy.functionspace import LagrangeFESpace as OLFE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

def meshpy2d(points, facets, h, hole_points=None, facet_markers=None, point_markers=None, meshtype='tri'):
    from meshpy.triangle import MeshInfo, build

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)

    mesh_info.set_holes(hole_points)

    mesh = build(mesh_info, max_volume=h**2)

    node = np.array(mesh.points, dtype=np.float64)
    cell = np.array(mesh.elements, dtype=np.int_)
    mesh = OTM(node,cell) 
    return mesh

tmr = timer()
next(tmr)

points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
        dtype=np.float64)
facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)
mm = IntervalMesh.from_circle_boundary([0.2, 0.2], 0.1, int(2*0.1*np.pi/0.01))
p = mm.entity('node')
f = mm.entity('cell')
points = np.append(points, p, axis=0)
facets = np.append(facets, f+4, axis=0)
fm = np.array([0, 1, 2, 3])
omesh = meshpy2d(points, facets, 0.01, hole_points=[[0.2, 0.2]], facet_markers=fm)

fkwargs = {'dtype': torch.float64, 'device': device}
ikwargs = {'dtype': torch.int, 'device': device}
node = torch.from_numpy(omesh.entity('node')).to(**fkwargs)
cell = torch.from_numpy(omesh.entity('cell')).to(**ikwargs)

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

pde = PDE()

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





