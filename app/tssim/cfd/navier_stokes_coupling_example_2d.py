import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

import matplotlib.pyplot as plt
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine
## Stokes model
from navier_stokes_mold_2d import Poisuille as PDE

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from fealpy.fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from fealpy.fem import VectorDiffusionIntegrator
from fealpy.fem import VectorViscousWorkIntegrator, PressWorkIntegrator
from fealpy.fem import ScalarConvectionIntegrator
from fealpy.fem import BilinearForm, MixedBilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import VectorSourceIntegrator, ScalarSourceIntegrator
from fealpy.fem import DirichletBC

# 参数设置
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解NS方程
        """)

parser.add_argument('--udegree',
        default=2, type=int,
        help='运动有限元空间的次数, 默认为 2 次.')

parser.add_argument('--pdegree',
        default=1, type=int,
        help='压力有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 5000 段.')

parser.add_argument('--T',
        default=10, type=float,
        help='演化终止时间, 默认为 5')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

parser.add_argument('--step',
        default=10, type=int,
        help='隔多少步输出一次')

parser.add_argument('--method',
        default='Netwon', type=str,
        help='非线性化方法')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
output = args.output
step = args.step
method = args.method
ns = 8

mu= 1
rho = 1
udim = 2
doforder = 'sdofs'

pde = PDE()
mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
smesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
uspace = LagrangeFiniteElementSpace(smesh,p=udegree)
pspace = LagrangeFiniteElementSpace(smesh,p=pdegree)
nuspace = LagrangeFESpace(mesh,p=2,doforder=doforder)
npspace = LagrangeFESpace(mesh,p=1,doforder=doforder)


u0 = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)

p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

Vbform0 = BilinearForm(nuspace)
Vbform0.add_domain_integrator(ScalarDiffusionIntegrator())
Vbform0.assembly()
A = Vbform0.get_matrix()

Vbform1 = MixedBilinearForm((npspace,), 2*(nuspace, ))
Vbform1.add_domain_integrator(PressWorkIntegrator()) #TODO: 命名
Vbform1.assembly()
B = Vbform1.get_matrix()
B1 = B[:B.shape[0]//2,:]
B2 = B[B.shape[0]//2:,:]

E = (1/dt)*uspace.mass_matrix()

errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0,3):

    # 下一个时间层t1
    t1 = tmesh.next_time_level()
    print("t1=",t1)

    Vbform2 = BilinearForm(nuspace)
    Vbform2.add_domain_integrator(ScalarConvectionIntegrator(c=u0)) 
    Vbform2.assembly()
    DD = Vbform2.get_matrix()
    
    D1,D2 = uspace.div_matrix(uspace)
    D = D1 * np.broadcast_to(u0[...,0],D1.shape)+\
        D2 * np.broadcast_to(u0[...,1],D1.shape) 
    
    print("asd",np.abs(D1).sum())
    print("asd",np.abs(D).sum())
    print(np.sum(np.abs(D-DD))) 
    M = bmat([[E+A+D,None,-B1],[None,E+A+D,-B2],[-B1.T,-B2.T,None]],format='csr')
    ''' 
    if method == 'Netwon' :
        A = bmat([[1/dt*M + mu*S+D1+D2+E1, E2, -C1],\
                [E3, 1/dt*M + mu*S +D1+D2+E4, -C2],\
                [C1.T, C2.T, None]], format='csr')
    elif method == 'Ossen':
        A = bmat([[1/dt*M + mu*S+D1+D2, None, -C1],\
                [None, 1/dt*M + mu*S +D1+D2, -C2],\
                [C1.T, C2.T, None]], format='csr')
    elif method == 'Eular':
        A = bmat([[1/dt*M + mu*S, None, -C1],\
                [None, 1/dt*M + mu*S, -C2],\
                [C1.T, C2.T, None]], format='csr')
    '''
    #右端项
    F = uspace.source_vector(pde.source,dim=udim) + E@u0
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]
    '''
    if method == 'Netwon' :
        b = 1/dt*fb1 + fb2
        b = np.hstack((b,[0]*pgdof))
    elif method == 'Ossen':
        b = 1/dt*fb1
        b = np.hstack((b,[0]*pgdof))
    elif method == 'Eular':
        b =  1/dt*fb1 - fb2
        b = np.hstack((b,[0]*pgdof))
    '''
    u_isBdDof = uspace.is_boundary_dof()
    #p_isBdDof = np.zeros(pgdof,dtype=np.bool)
    p_isBdDof = pspace.is_boundary_dof(threshold=pde.is_p_boundary)
    
    x = np.zeros(gdof,np.float64)
    ipoint = uspace.interpolation_points()
    uso = pde.u_dirichlet(ipoint)
    x[0:ugdof][u_isBdDof] = uso[:,0][u_isBdDof]
    x[ugdof:2*ugdof][u_isBdDof] = uso[u_isBdDof][:,1]
    ipoint = pspace.interpolation_points()
    pso = pde.p_dirichlet(ipoint)
    x[-pgdof:][p_isBdDof] = pso[p_isBdDof]

    isBdDof = np.hstack([u_isBdDof, u_isBdDof, p_isBdDof])
    
    FF -= M@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    M = T@M@T + Tbd
    FF[isBdDof] = x[isBdDof]

    x[:] = spsolve(M, FF)
    u1[:, 0] = x[:ugdof]
    u1[:, 1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    uc1 = pde.velocity(smesh.node)
    NN = smesh.number_of_nodes()
    uc2 = u1[:NN]
    up1 = pde.pressure(smesh.node)
    up2 = p1[:NN]
   
    errorMatrix[0,i] = uspace.integralalg.L2_error(pde.velocity,u1)
    errorMatrix[1,i] = pspace.integralalg.error(pde.pressure,p1)
    errorMatrix[2,i] = np.abs(uc1-uc2).max()
    errorMatrix[3,i] = np.abs(up1-up2).max()

    u0[:] = u1 

    tmesh.advance()
print(np.sum(np.abs(u1)))
'''
print("uL2:",errorMatrix[2,-1])
print("pL2:",errorMatrix[1,-1])
print("umax:",errorMatrix[2,-1])
print("pmax:",errorMatrix[3,-1])
fig1 = plt.figure()
node = smesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = smesh.number_of_nodes()
u = u1[:NN]
ux = tuple(u[:,0])
uy = tuple(u[:,1])

o = ux
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.colorbar(sm)
plt.show()
'''
