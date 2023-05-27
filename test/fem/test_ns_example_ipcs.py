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
nt=50
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
Vbform0 = BilinearForm(2*(uspace,))
Vbform0.add_domain_integrator(VectorMassIntegrator(rho/dt))
Vbform0.assembly()
A1 = Vbform0.get_matrix()

Vbform1 = BilinearForm(2*(uspace,))
Vbform1.add_domain_integrator(NSOperatorIntegrator(mu))
Vbform1.assembly()
A2 = Vbform1.get_matrix()
A = A1+A2
print("第一个方程左端：",np.sum(np.abs(A.toarray())))

#第二个
Sbform = BilinearForm(pspace)
Sbform.add_domain_integrator(ScalarDiffusionIntegrator(c=1))
Sbform.assembly()
B = Sbform.get_matrix()
print("第二个方程的左端：",np.sum(np.abs(B.toarray())))

#第三个
Vbform2 = BilinearForm(2*(uspace,))
Vbform2.add_domain_integrator(VectorMassIntegrator(c=1))
Vbform2.assembly()
C = Vbform2.get_matrix()
print("第三个方程的左端:",np.sum(np.abs(C.toarray())))

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

@barycentric
def f(bcs,index):
    b1 = u0(bcs,index)
    b2 = np.einsum('idc,idjc->ijc',u0(bcs,index),u0.grad_value(bcs,index))
    return rho/dt*b1+b2

qf = mesh.integrator(4)
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = uspace.cellmeasure
uphi = uspace.basis(bcs)
#fbb1 = np.einsum('i,ijk,imj,j -> jkm',ws,uphi,u0(bcs),cellmeasure)
fbb1 = np.einsum('i,ijk,imj,j -> jkm',ws,uphi,f(bcs,np.s_[:]),cellmeasure)
ugdof = uspace.number_of_global_dofs()
ucell2dof = uspace.cell_to_dof()
fb1 = np.zeros((ugdof,2))
np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)
print(np.sum(np.abs(fb1)))


lform = LinearForm(2*(uspace,))
lform.add_domain_integrator(VectorSourceIntegrator(f))
lform.assembly()
b1 = lform.get_vector()
print(np.sum(np.abs(b1)))

'''
for i in range(nt):
    t1 = timeline.next_time_level()
    print("t=", t1)
    
    b10 = A@u0
    
    @barycentric
    def f1(bcs,index):
        b2 = np.einsum('idc,idjc->ijc',u0(bcs,index),u0.grad_value(bcs,index))
        return rho*b2

    lform = LinearForm(2*(uspace,))
    lform.add_domain_integrator(VectorSourceIntegrator(f2))
    lform.assembly()
    b12 = lform.get_vector()

    b11 = A1@u0 
    b13 = A2@u0
    b1 =  b11-b12-b13+

    
    b21 = B@p0
    b22 = 

    
    @barycentric
    def f1(bcs,index):
        c1 = us(bcs,index)
        c2 = np.einsum('idc,idjc->ijc',p0.grad_value(bcs,index),p1.grad_value(bcs,index))
        c3 = np.einsum('idc,idjc->ijc',p0.grad_value(bcs,index),p1.grad_value(bcs,index))
        return rho*b2

    lform = LinearForm(2*(uspace,))
    lform.add_domain_integrator(VectorSourceIntegrator(f2))
    lform.assembly()
        c2 = 
        return rho*b2


    timeline.advance() 
'''
