#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: poisuille_ossen.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 02 Jul 2024 03:03:56 PM CST
	@bref 
	@ref 
'''  
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat
from scipy.sparse import spdiags, bmat

from fealpy.mesh import TriangleMesh
from fealpy.pde.navier_stokes_equation_2d import Poisuille
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.cfd import NSFEMSolver
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DirichletBC

from fealpy.decorator import barycentric
from fealpy.utils import timer
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator
from fealpy.fem import ScalarConvectionIntegrator, ScalarSourceIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import PressWorkIntegrator
from fealpy.fem import MixedBilinearForm, BilinearForm, LinearForm
#get basic parameters
ns = 16
udegree = 2
pdegree = 1
T = 10
nt = 1000
pde = Poisuille()
rho = pde.rho
mu = pde.mu
q = 5

time = timer()
next(time)
mesh = TriangleMesh.from_box(pde.domain(), nx = ns, ny = ns)
timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt
uspace = LagrangeFESpace(mesh, p=udegree, doforder='sdofs')
pspace = LagrangeFESpace(mesh, p=pdegree, doforder='sdofs')

u0 = uspace.function(dim = 2)
u1 = uspace.function(dim = 2)
p1 = pspace.function()

source = uspace.interpolate(pde.source, dim=2)

pgdof = pspace.number_of_global_dofs()
ugdof = uspace.number_of_global_dofs()
gdof = 2*ugdof + pgdof 

uso = uspace.interpolate(pde.velocity,2)
pso = pspace.interpolate(pde.pressure)

bform = BilinearForm(uspace)
bform.add_domain_integrator(ScalarMassIntegrator(c=rho/dt,q=q))
M = bform.assembly()

bform = BilinearForm(uspace)
bform.add_domain_integrator(ScalarDiffusionIntegrator(c=mu,q=q))
S = bform.assembly()

bform = MixedBilinearForm((pspace,), 2*(uspace,)) 
bform.add_domain_integrator(PressWorkIntegrator(q=q)) 
AP = bform.assembly()

#边界处理
xx = np.zeros(gdof, np.float64)

is_u_bdof = uspace.is_boundary_dof()
is_p_bdof = pspace.is_boundary_dof()

xx[:ugdof][is_u_bdof] = uso[0,:][is_u_bdof]
xx[ugdof:2*ugdof][is_u_bdof] = uso[1,:][is_u_bdof]
xx[2*ugdof:][is_p_bdof] = pso[is_p_bdof]

isBdDof = np.hstack([is_u_bdof,is_u_bdof,is_p_bdof])
bdIdx = np.zeros(gdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)

errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0,1):
    t1 = timeline.next_time_level()
    print("time=", t1)
    
    @barycentric
    def concoef(bcs, index):
        return rho*u0(bcs,index)
    
    bform = BilinearForm(uspace)
    bform.add_domain_integrator(ScalarConvectionIntegrator(concoef, q=q))
    C = bform.assembly()
     
    A0 = bmat([[M+S+C, None],\
              [None, M+S+C]], format='csr')
    A = bmat([[A0, -AP],\
             [-AP.T, None]], format='csr')
    
    @barycentric
    def bcoef(bcs, index):
        return 1/dt*rho*u0(bcs, index)
    
    b0 = M@u0[0,:] 
    b1 = M@u0[1,:]
    b2 = [0]*pgdof 
    b = np.hstack((b0,b1,b2))
    #print("A",np.sum(np.abs(A)))
    
    #A = T@A + Tbd
    #b[isBdDof] = xx[isBdDof]
    b -= A@xx
    A = T@A@T + Tbd
    b[isBdDof] = xx[isBdDof]
    x = spsolve(A,b)
     
    u1[0,:] = x[:ugdof]
    u1[1,:] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    u0[:] = u1

    errorMatrix[0,i] = mesh.error(pde.velocity, u1)
    errorMatrix[1,i] = mesh.error(pde.pressure, p1)
    errorMatrix[2,i] = np.abs(uso-u1).max()
    errorMatrix[3,i] = np.abs(pso-p1).max()
    print(errorMatrix[2,i])
    print(errorMatrix[3,i])
    timeline.advance()
