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
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.fem import DiffusionIntegrator, MassIntegrator, ConvectionIntegrator
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.decorator import cartesian, barycentric
from fealpy.fem import LinearForm
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import SourceIntegrator
from fealpy.timeintegratoralg import UniformTimeLine
from scipy.sparse import csr_matrix,bmat

T=2
nt=100
ns = 20
p = 2
mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
space = LagrangeFESpace(mesh,p=p,doforder='sdofs')
oldspace = LagrangeFiniteElementSpace(mesh,p=p)
timeline = UniformTimeLine(0,T,nt)
dt = timeline.dt

bform = BilinearForm(2*(space,))
#bform.add_domain_integrator(MassIntegrator())
#bform.add_domain_integrator(DiffusionIntegrator())
bform.add_domain_integrator(LinearElasticityOperatorIntegrator(lam=0,mu=0.5))
bform.assembly()
A = bform.get_matrix()


qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
ugphi = oldspace.grad_basis(bcs)
ugdof = oldspace.number_of_global_dofs()
ucell2dof = oldspace.cell_to_dof()
E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)
E2 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,1],cellmeasure)
E3 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,0],cellmeasure)
I = np.broadcast_to(ucell2dof[:,:,None],shape = E0.shape)
J = np.broadcast_to(ucell2dof[:,None,:],shape = E0.shape)
E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E = bmat([[E00,E01],[E10,E11]])

#AA = oldspace.mass_matrix()
AA = oldspace.stiff_matrix()
AA = bmat([[AA,None],[None,AA]])
print(np.sum(np.abs(E-A)))

