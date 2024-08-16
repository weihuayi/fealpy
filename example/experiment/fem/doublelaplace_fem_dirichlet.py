import argparse
import sympy as sp
from matplotlib import pyplot as plt

from fealpy.experimental import logger
logger.setLevel('WARNING')
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import CmConformingFESpace2d 
from fealpy.experimental.fem import BilinearForm 
from fealpy.experimental.fem.grad_m_integrator import gradmIntegrator
from fealpy.experimental.fem import LinearForm, ScalarSourceIntegrator
from fealpy.experimental.fem import DirichletBC
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.solver import cg
from fealpy.experimental.pde.biharmonic_triharmonic_2d import DoubleLaplacePDE, get_flist
from fealpy.utils import timer
from fealpy.decorator import barycentric
from scipy.sparse.linalg import spsolve
from fealpy.experimental import logger
logger.setLevel('INFO')
## 参数解析
parser = argparse.ArgumentParser(description=
        """
        光滑元有限元方法求解双调和方程
        """)

parser.add_argument('--degree',
        default=5, type=int,
        help='光滑有限元空间的次数, 默认为 5 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=5, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--meshtype',
        default='tri', type=str,
        help='默认网格为三角形网格')

args = parser.parse_args()


bm.set_backend(args.backend)
p = args.degree
n = args.n
meshtype = args.meshtype
maxit = args.maxit

tmr = timer()
next(tmr)
x = sp.symbols('x')
y = sp.symbols('y')
u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x))**2
pde = DoubleLaplacePDE(u) 
ulist = get_flist(u)[:3]
mesh = TriangleMesh.from_box([0,1,0,1], n, n)

errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega,0}$',
             '$||\\nabla^2 u - \\nabla^2 u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((3, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')

for i in range(maxit):
    node = mesh.entity('node')
    isCornerNode = bm.zeros(len(node),dtype=bm.bool)
    for n in bm.array([[0,0],[1,0],[0,1],[1,1]], dtype=bm.float64):
        isCornerNode = isCornerNode | (bm.linalg.norm(node-n    [None, :], axis=1)<1e-10)


    space = CmConformingFESpace2d(mesh, p, 1, isCornerNode)
    

    uh = space.function()

    bform = BilinearForm(space)
    coef = 1
    integrator = gradmIntegrator(m=2, coef=1, q=p+4)
    bform.add_integrator(integrator)
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source, q=p+4))

    A = bform.assembly()
    F = lform.assembly()
    tmr.send(f'第{i}次矩组装时间')



    gdof = space.number_of_global_dofs()
    bc1 = DirichletBC(space, gd = ulist)
    A, F = bc1.apply(A, F)  
    tmr.send(f'第{i}次边界处理时间')
    #A = A.to_dense()
    #uh[:] = bm.linalg.solve(A, F)
    #A = A.toarray()
    #print(bm.linalg.cond(A))
    #import ipdb
    #ipdb.set_trace()
    #import scipy.sparse
    #A = scipy.sparse.csr_matrix(A)
    #uh[:] = spsolve(A, F)
    
    uh[:] = cg(A, F, maxiter=400000, atol=1e-14, rtol=1e-14)
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
    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')
next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("order : ", bm.log2(errorMatrix[1,:-1]/errorMatrix[1,1:]))
print("order : ", bm.log2(errorMatrix[2,:-1]/errorMatrix[2,1:]))


