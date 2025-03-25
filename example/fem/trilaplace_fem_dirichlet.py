import argparse
import sympy as sp
from matplotlib import pyplot as plt

from fealpy import logger
logger.setLevel('WARNING')
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import CmConformingFESpace2d 
from fealpy.fem import BilinearForm 
from fealpy.fem.mthlaplace_integrator import MthLaplaceIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.backend import backend_manager as bm
from fealpy.solver import cg
from fealpy.pde.biharmonic_triharmonic_2d import TripleLaplacePDE, get_flist
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
        光滑元有限元方法求解双调和方程
        """)

parser.add_argument('--degree',
        default=9, type=int,
        help='光滑有限元空间的次数, 默认为 9 次.')

parser.add_argument('--n',
        default=1, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

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
#u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x))**4
#u = (sp.sin(4*sp.pi*y)*sp.sin(4*sp.pi*x))**6
#u = (sp.sin(2*y)*sp.sin(2*x))
#u = x**3*y**4
#u = x**2*(x-1)**5*y+2*x*y**2*(y-1)**5
u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x))
pde = TripleLaplacePDE(u) 
ulist = get_flist(u, device=device)
mesh = TriangleMesh.from_box([0,1,0,1], n, n, device=device)

ikwargs = bm.context(mesh.cell)
fkwargs = bm.context(mesh.node)

NDof = bm.zeros(maxit, **ikwargs)

errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega,0}$',
             '$||\\nabla^2 u - \\nabla^2 u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((3, maxit), **fkwargs)
tmr.send('网格和pde生成时间')

for i in range(maxit):
    node = mesh.entity('node')
    isCornerNode = bm.zeros(len(node),dtype=bm.bool, device=device)
    for n in bm.array([[0,0],[1,0],[0,1],[1,1]], **fkwargs):
        isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)



    space = CmConformingFESpace2d(mesh, p, 2, isCornerNode)
    
    tmr.send(f'第{i}次空间生成时间')

    uh = space.function()

    bform = BilinearForm(space)
    coef = 1
    integrator = MthLaplaceIntegrator(m=3, coef=1, q=p+4)
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
    #import ipdb
    #ipdb.set_trace()
    A, F = bc1.apply(A, F)  
    tmr.send(f'第{i}次边界处理时间')
    #A = A.to_scipy()

    #from numpy.linalg import cond
    #print(gdof)
    #print(cond(A.toarray()))
    #A = coo_matrix(A)
    #A = csr_matrix((A.values(), A.indices()),A.shape)
    #uh[:] = bm.tensor(spsolve(A, F))
    uh[:] = spsolve(A, F, "scipy")
    
    #uh[:] = cg(A, F, maxiter=400000, atol=1e-14, rtol=1e-14)
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

