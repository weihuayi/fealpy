import argparse
import numpy as np
import sympy as sp
import sys
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat, construct
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

#import fealpy.mesh.MeshFactory as mf
from fealpy.mesh import TriangleMesh

## function space
from fealpy.functionspace.HuZhangFiniteElementSpace2D import HuZhangFiniteElementSpace
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace

#linear elasticity model
from fealpy.pde.linear_elasticity_model import QiModel3d, PolyModel3d, Model2d, HuangModel2d
from fealpy.pde.linear_elasticity_model2D import GenLinearElasticitymodel2D

#solver 
#from fealpy.solver.fast_solver import LinearElasticityHZFEMFastSolve

## error anlysis tool
from fealpy.tools.show import showmultirate


##  参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上用胡张元求解线弹性力学问题
        """)

parser.add_argument('--degree',
        default=3, type=int,
        help='Lagrange 有限元空间的次数, 默认为 3 次.')

parser.add_argument('--nrefine',
        default=2, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--bdtype',
        default='displacement', type=str,
        help='边界条件, 默认为位移边界')

args = parser.parse_args()

degree = args.degree
nrefine = args.nrefine
maxit = args.maxit
bdtype = args.bdtype


#由位移生成pde模型
#E = 1e3
#nu = 0.3
#lam = E*nu/((1+nu)*(1-2*nu))
#mu = E/(2*(1+nu))


pi = sp.pi
sin = sp.sin
cos = sp.cos
exp = sp.exp
ln = sp.ln

#给定应力算例
lam = 1
mu = 1

#一般pde算例
x = sp.symbols('x0:2')
#u = [0.01*(1-x[0]),0]
#u = [sin(2*pi*x[0])*sin(2*pi*x[1]),sin(2*pi*x[0])*sin(2*pi*x[1])]
#u = [exp(x[0]+x[1]),exp(x[0]-x[1])]
#u = [x[0],x[1]]
#u = [0,(x[0]-1)**7*(x[1]-1)**7]
u = [-(1-x[0])*ln(1.5-x[0]),-(1-x[0])*ln(1.5-x[1])]
#u = [sin(2*pi*x[0])*sin(2*pi*x[1]),sin(2*pi*x[0])*sin(2*pi*x[1])]


if bdtype == 'displacement': 
        pde = GenLinearElasticitymodel2D(u,x,lam=lam,mu=mu,
                Dirichletbd_n='(x0==1)|(x0==0)|(x1==0)|(x1==1)',
                Dirichletbd_t='(x0==1)|(x0==0)|(x1==0)|(x1==1)')

elif bdtype =='stress_and_displacement':
        pde = GenLinearElasticitymodel2D(u,x,lam=lam,mu=mu,
                Dirichletbd_n='(x1==0)|(x1==1)',Dirichletbd_t='(x1==0)|(x1==1)',
                Neumannbd_nn='(x0==1)|(x0==0)',Neumannbd_nt='(x0==1)|(x0==0)')

elif bdtype =='stress_and_displacement_corner_point':
        pde = GenLinearElasticitymodel2D(u,x,lam=lam,mu=mu,
                Dirichletbd_n='(x0==1)|(x1==1)',Dirichletbd_t='(x0==1)|(x1==1)',
                Neumannbd_nn='(x0==0)|(x1==0)',Neumannbd_nt='(x0==0)|(x1==0)')
                        



#mesh = mf.boxmesh2d(pde.domain(),nx=1,ny=1,meshtype='tri')
mesh = TriangleMesh.from_box(pde.domain(), nx=1, ny=1)
mesh.uniform_refine(nrefine)



errorType = ['$||\sigma - \sigma_h ||_{0}$',
        '$||div(\sigma - \sigma_h)||_{0}$',
        '$||u - u_h||_{0}$'
        ]
Ndof = np.zeros((maxit,))
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float64)
gdim = 2

for i in range(maxit):
        print("The {}-th computation:".format(i))

        tspace = HuZhangFiniteElementSpace(mesh, degree)
        vspace = LagrangeFiniteElementSpace(mesh, degree-1, spacetype='D')

        tgdof = tspace.number_of_global_dofs()
        vgdof = vspace.number_of_global_dofs()

        sh = tspace.function()
        uh = vspace.function(dim=gdim)

        #M = tspace.compliance_tensor_matrix(mu=pde.mu,lam=pde.lam)
        M = tspace.parallel_compliance_tensor_matrix(mu=pde.mu,lam=pde.lam)

        #B0,B1 = tspace.div_matrix(vspace)
        B0,B1 = tspace.parallel_div_matrix(vspace)
        #print(np.max(np.abs(B0-B10)))

        F1 =  -vspace.source_vector(pde.source,dim=gdim)


        #边界处理
        F0 = tspace.set_nature_bc(pde.dirichlet,threshold=pde.is_dirichlet_boundary) #此处以位移边界为dirichlet边界

        isBDdof = tspace.set_essential_bc(sh, pde.neumann,M,B0,B1,F0, threshold=pde.is_neumann_boundary)#以应力边界为neumann边界


        F0 -= M@sh
        F0[isBDdof] = sh[isBDdof]
        F1[:,0] -= B0@sh 
        F1[:,1] -= B1@sh
        

        bdIdx = np.zeros(tgdof, dtype=np.int_)
        bdIdx[isBDdof] = 1
        Tbd = spdiags(bdIdx,0,tgdof,tgdof)
        T = spdiags(1-bdIdx,0,tgdof,tgdof)
        M = T@M@T + Tbd
        B0 = B0@T
        B1 = B1@T
        


        #求解
        FF = np.r_[F0,F1.T.reshape(-1)]
        AA = bmat([[M, B0.transpose(), B1.transpose()],[B0, None, None],[B1,None,None]],format='csr')
        x = spsolve(AA,FF)

    #B = construct.vstack([B0,B1],format='csr')
    #A = [M,B]
    #F = [F0,F1]
    #Fast_slover = LinearElasticityHZFEMFastSolve(A,F,vspace)
    #x = Fast_slover.solve()




        sh[:] = x[:tgdof]
        uh[:,0] = x[tgdof:tgdof+vgdof]
        uh[:,1] = x[tgdof+vgdof:]
 
        gdof = tgdof+gdim*vgdof
        Ndof[i] = gdof

        #有限元误差
        errorMatrix[0,i] = tspace.integralalg.error(pde.stress,sh.value)
        errorMatrix[1,i] = tspace.integralalg.error(pde.div_stress,sh.div_value)
        errorMatrix[2,i] = vspace.integralalg.error(pde.displacement,uh.value)        
        if i < maxit - 1:
                mesh.uniform_refine()

       
print('Ndof:', Ndof)
print('error:', errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()













        
