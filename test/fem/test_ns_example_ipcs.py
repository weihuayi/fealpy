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
T=10
nt=50
ns = 16
rho = 1
mu = 1

mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
timeline = UniformTimeLine(0,T,nt)
uspace = LagrangeFESpace(mesh,p=2,doforder='sdofs')
pspace = LagrangeFESpace(mesh,p=1,doforder='sdofs')
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
