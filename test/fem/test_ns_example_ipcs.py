#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_ns_example.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年04月11日 星期二 18时42分28秒
	@bref 
	@ref 
'''  
import numpy as np
from scipy.sparse import bmat,csr_matrix,hstack,vstack,spdiags
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine

from scipy.sparse import csr_matrix,hstack,vstack,spdiags
from fealpy.fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from fealpy.fem import NSOperatorIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import VectorSourceIntegrator

from fealpy.decorator import cartesian, barycentric
T=10
nt=5
ns = 16
rho = 1
mu = 1
doforder = 'sdofs'

mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
timeline = UniformTimeLine(0,T,nt)
uspace = LagrangeFESpace(mesh,p=2,doforder=doforder)
pspace = LagrangeFESpace(mesh,p=1,doforder=doforder)
dt = timeline.dt

# 第一个
Vbform = BilinearForm(2*(uspace,))
Vbform.add_domain_integrator(VectorMassIntegrator(rho/dt))
Vbform.add_domain_integrator(NSOperatorIntegrator(mu))
#Vbform.add_domain_integrator(NSOperatorIntegrator(1,3))
Vbform.assembly()
A = Vbform.get_matrix()
print(np.sum(np.abs(A.toarray())))

#第二个
Sbform = BilinearForm(pspace)
Sbform.add_domain_integrator(ScalarDiffusionIntegrator(c=1))
Sbform.assembly()
B = Sbform.get_matrix()
print(np.sum(np.abs(B.toarray())))

#第三个
Vbform1 = BilinearForm(2*(uspace,))
Vbform1.add_domain_integrator(VectorMassIntegrator(c=1))
Vbform1.assembly()
C = Vbform1.get_matrix()
print(np.sum(np.abs(C.toarray())))

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)

p0 = pspace.function()
p1 = pspace.function()

@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = 4*y*(1-y)
    u[..., 1] = 0
    return u

u0 = uspace.interpolate(velocity_field, dim=2)
'''
@barycentric
def f(bcs,index):
    b1 = u0(bcs,index)
    b2 = np.einsum('imk,ijmk->ijk',u0(bcs,index),u0.grad_value(bcs,index))
    return b1+b2 
'''

qf = mesh.integrator(4)
bcs,ws = qf.get_quadrature_points_and_weights()
print(u0(bcs).shape)
print(u0.grad_value(bcs).shape)
#b2 = np.einsum('imk,ijmk->ijk',u0(bcs),u0.grad_value(bcs))

#lform = LinearForm(2*(uspace,))
#lform.add_domain_integrator(VectorSourceIntegrator(f))
#lform.assembly()
#b1 = lform.get_vector()
'''
fuu = u0(bcs)
fbb1 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,fuu,cellmeasure)
fb1 = np.zeros((ugdof,2))
np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)
print(fb1.shape)
print(np.sum(np.abs(fb1)))
fb1 = (rho/dt)*fb1

fgu = u0.grad_value(bcs)
fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
fb2 = np.zeros((ugdof,2))
np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
'''


'''
for i in range(nt):
    t1 = timeline.next_time_level()
    print("t=", t1)
    


    timeline.advance() 
'''
