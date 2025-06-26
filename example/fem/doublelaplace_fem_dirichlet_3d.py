import argparse
import sympy as sp
from matplotlib import pyplot as plt

from fealpy import logger
logger.setLevel('WARNING')
from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace.cm_conforming_fe_space_3d import CmConformingFESpace3d 
from fealpy.fem import BilinearForm 
from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.backend import backend_manager as bm
from fealpy.pde.biharmonic_triharmonic_3d import DoubleLaplacePDE, get_flist
from fealpy.utils import timer
from fealpy.decorator import barycentric
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from fealpy import logger
from fealpy.solver import spsolve
logger.setLevel('INFO')
## 参数解析
parser = argparse.ArgumentParser(description=
        """
        光滑元有限元方法求解三维双调和方程
        """)

parser.add_argument('--degree',
        default=9, type=int,
        help='光滑有限元空间的次数, 默认为 9 次.')

parser.add_argument('--n',
        default=1, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 3 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--device',
        default='cpu', type=str,
        help='默认gpu计算')

args = parser.parse_args()


bm.set_backend(args.backend)
#device = "cuda"
p = args.degree
n = args.n
maxit = args.maxit
device = args.device

tmr = timer()
next(tmr)
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
#u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x))**4
#u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x))**2
u = sp.sin(4*x)*sp.sin(4*y)*sp.sin(4*z)
pde = DoubleLaplacePDE(u, device=device) 
ulist = get_flist(u, device=device)
mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], n, n, n, device=device)

ikwargs = bm.context(mesh.cell)
fkwargs = bm.context(mesh.node)

NDof = bm.zeros(maxit, **fkwargs)

errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega,0}$',
             '$||\\nabla^2 u - \\nabla^2 u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((3, maxit), **fkwargs)
errorMatrix1 = bm.zeros((3, maxit), **fkwargs)
tmr.send('网格和pde生成时间')

for i in range(maxit):
    space = CmConformingFESpace3d(mesh, p, 1)
    
    tmr.send(f'第{i}次空间生成时间')

    uh = space.function()

    bform = BilinearForm(space)
    coef = 1
    integrator = MthLaplaceIntegrator(m=2, coef=1, q=p+4,
                                      method='without_numerical_integration')
    bform.add_integrator(integrator)
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source, q=p+4))

    A = bform.assembly()
    #print(space.number_of_global_dofs())
    F = lform.assembly()
    tmr.send(f'第{i}次矩组装时间')



    gdof = space.number_of_global_dofs()
    NDof[i] = 1/4/2**i
    bc1 = DirichletBC(space, gd = ulist)
    A, F = bc1.apply(A, F)  
    tmr.send(f'第{i}次边界处理时间')
    uh[:] = spsolve(A, F, "mumps")
    uh1 = space.function()
    uI = space.interpolation(ulist)
    uh1[:] = uI- uh[:]
    
    tmr.send(f'第{i}次求解器时间')

    @barycentric
    def ugval(p):
        return space.grad_m_value(uh, p, 1)

    @barycentric
    def ug2val(p):
        return space.grad_m_value(uh, p, 2)
    errorMatrix[0, i] = mesh.error(pde.solution, uh)
    errorMatrix[1, i] = mesh.error(pde.gradient, ugval)
    errorMatrix[2, i] = mesh.error(pde.hessian, ug2val)
    @barycentric
    def ugval1(p):
        return space.grad_m_value(uh1, p, 1)

    @barycentric
    def ug2val1(p):
        return space.grad_m_value(uh1, p, 2)
    errorMatrix1[0, i] = mesh.error(uh1, 0)
    errorMatrix1[1, i] = mesh.error(ugval1, 0)
    errorMatrix1[2, i] = mesh.error(ug2val1, 0)
    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("order : ", bm.log2(errorMatrix[1,:-1]/errorMatrix[1,1:]))
print("order : ", bm.log2(errorMatrix[2,:-1]/errorMatrix[2,1:]))
print("最终误差",errorMatrix1)                                                  
print("order : ", bm.log2(errorMatrix1[0,:-1]/errorMatrix1[0,1:]))              
print("order : ", bm.log2(errorMatrix1[1,:-1]/errorMatrix1[1,1:]))              
print("order : ", bm.log2(errorMatrix1[2,:-1]/errorMatrix1[2,1:]))
import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
fig = plt.figure()                                                              
axes = fig.gca()                                                                
linetype = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x','r-.x']                 
c = np.polyfit(np.log(NDof),np.log(errorMatrix1[0]),1)                          
print(c)                                                                           
axes.loglog(NDof,errorMatrix1[0],linetype[0],label =                            
            '$||u-u_h||_{\\Omega,0}=O(h^{%0.4f})$'%(c[0]))                      
c = np.polyfit(np.log(NDof),np.log(errorMatrix1[1]),1)                          
axes.loglog(NDof,errorMatrix1[1],linetype[1],label =                            
            '$||\\nabla u-\\nabla u_h||_{\\Omega,0}=O(h^{%0.4f})$'%(c[0]))      
c = np.polyfit(np.log(NDof),np.log(errorMatrix1[2]),1)                          
axes.loglog(NDof,errorMatrix1[2],linetype[2],label =                            
            '$||\\nabla^2 u-\\nabla^2 u_h||_{\\Omega,0}=O(h^{%0.4f})$'%(c[0]))  
axes.legend()
#filename = f'cm.png'
#plt.savefig(filename)

plt.show()
