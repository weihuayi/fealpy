
import argparse

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.experimental import logger
logger.setLevel('WARNING')
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import SemilinearForm
from fealpy.experimental.fem import ScalarSemilinearMassIntegrator, ScalarSemilinearDiffusionIntegrator
from fealpy.experimental.fem import ScalarSourceIntegrator
from fealpy.experimental.pde.semilinear_2d import SemilinearData
from fealpy.utils import timer
from fealpy.decorator import barycentric


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次有限元方法求解半线性方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--meshtype',
        default='tri', type=str,
        help='默认网格为三角形网格')

args = parser.parse_args()
args.backend = 'pytorch'
bm.set_backend(args.backend)
p = args.degree
n = args.n
meshtype = args.meshtype
maxit = args.maxit

tmr = timer()
next(tmr)

domain = [0, 1, 0, 2]
pde = SemilinearData(domain)
mesh = TriangleMesh.from_box(domain, nx=n, ny=n)

tol = 1e-14
NDof = bm.zeros(maxit, dtype=bm.int64)
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')
def kernel_func_reaction(u):
    return u**3

def grad_kernel_func_reaction(u):
    return 3*u**2

def kernel_func_diffusion(u):
    return u

def grad_kernel_func_diffusion(u):
    return bm.ones_like(u)

def diffusion_coef(p):
    return pde.diffusion_coefficient(p)

def reaction_coef(p):
    return pde.reaction_coefficient(p)

reaction_coef.kernel_func = kernel_func_reaction
diffusion_coef.kernel_func = kernel_func_diffusion

if bm.backend_name == 'numpy':
    reaction_coef.grad_kernel_func = grad_kernel_func_reaction
    diffusion_coef.grad_kernel_func = grad_kernel_func_diffusion
    
for i in range(maxit):
    #定义函数空间
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    tmr.send(f'第{i}次空间时间')

    u0 = space.function()
    du = space.function()
    diffusion_coef.uh = u0
    reaction_coef.uh = u0
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)[-1]
    isIDof = ~isDDof

    #定义积分子
    D = ScalarSemilinearDiffusionIntegrator(diffusion_coef, q=p+2)
    M = ScalarSemilinearMassIntegrator(reaction_coef, q=p+2)
    f = ScalarSourceIntegrator(pde.source, q=p+2)

    while True:
        #矩阵组装
        sform = SemilinearForm(space)
        sform.add_integrator([D, M])
        sform.add_integrator(f)
        A, F = sform.assembly()
        tmr.send(f'第{i}次矩组装时间')
        
        #求解增量
        A = csr_matrix((A.values(), A.indices()), A.shape)
        du[isIDof] = bm.tensor(spsolve(A[isIDof, :][:, isIDof], F[isIDof]).reshape(-1), dtype=du[isIDof].dtype)
        u0 += du
        tmr.send(f'第{i}次求解器时间')

        #清除积分子缓存
        D.clear()
        M.clear()

        #计算误差
        err = bm.max(bm.abs(du.array))
        if err < tol:
            break

    @barycentric
    def ugval(p):
        return space.grad_value(u0, p)

    errorMatrix[0, i] = mesh.error(pde.solution, u0, q=p+2)
    errorMatrix[1, i] = mesh.error(pde.gradient, ugval, q=p+2)
    if i < maxit-1:
        mesh.uniform_refine()
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)
print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])
