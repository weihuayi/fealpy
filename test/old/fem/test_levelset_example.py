#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_levelset_example.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年04月11日 星期二 11时29分47秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.fem import DiffusionIntegrator, MassIntegrator, ConvectionIntegrator
from fealpy.decorator import cartesian, barycentric
from fealpy.fem import LinearForm
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import SourceIntegrator
from fealpy.timeintegratoralg import UniformTimeLine
from scipy.sparse import bmat,csr_matrix,hstack,vstack,spdiags
from mumps import DMumpsContext


domain = [0, 1, 0, 1]
T=2
nt=100
ns = 20
p = 2
mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
space = LagrangeFESpace(mesh,p=p)
timeline = UniformTimeLine(0,T,nt)
dt = timeline.dt


@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@cartesian
def pic(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

ips = space.interpolation_points()
uh = space.function(array=pic(ips))

bform = BilinearForm(space)
bform.add_domain_integrator(MassIntegrator())
bform.add_domain_integrator(ConvectionIntegrator(c=velocity_field, a=dt, q=4))
bform.assembly()
A = bform.get_matrix()

ctx = DMumpsContext()
ctx.set_silent()
fname = '/home/wpx/result/ls/test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['uh'] = uh
mesh.to_vtk(fname=fname)

for i in range(nt):
    
    t1 = timeline.next_time_level()
    print("t=",t1)
    
    lform = LinearForm(space)
    lform.add_domain_integrator(SourceIntegrator(uh))
    lform.assembly()
    b = lform.get_vector()

    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)

    uh[:] = x
    fname = '/home/wpx/result/ls/test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['uh'] = uh
    mesh.to_vtk(fname=fname)
    timeline.advance()
ctx.destroy()

