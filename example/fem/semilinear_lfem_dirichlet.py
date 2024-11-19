
import argparse
import matplotlib.pyplot as plt

from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import NonlinearForm
from fealpy.fem import ScalarNonlinearMassIntegrator, ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.pde.semilinear_2d import SemilinearData
from fealpy.solver import cg
from fealpy.utils import timer
from fealpy.decorator import barycentric


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在三角形网格上使用任意次有限元方法求解半线性方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--n',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=6, type=int,
        help='默认网格加密求解的次数, 默认加密求解 6 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--device',
        default='cuda', type=str,
        help='默认gpu上运行')


args = parser.parse_args()
if args.backend =='pytorch':
    bm.set_default_device(args.device)

p = args.degree
n = args.n
maxit = args.maxit

tmr = timer()
next(tmr)

domain = [0, 1, 0, 2]
pde = SemilinearData(domain)
mesh = TriangleMesh.from_box(domain, nx=n, ny=n, device=args.device)

tol = 1e-14
NDof = bm.zeros(maxit, dtype=bm.int64, device=args.device)
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64, device=args.device)
tmr.send('网格和pde生成时间')

def diffusion_coef(p, **args):
    return pde.diffusion_coefficient(p)

def reaction_coef(p, **args):
    return pde.reaction_coefficient(p)

def kernel_func_reaction(u):
    return u**3

reaction_coef.kernel_func = kernel_func_reaction

if bm.backend_name == 'numpy':
    def grad_kernel_func_reaction(u):
        return 3*u**2

    reaction_coef.grad_kernel_func = grad_kernel_func_reaction

for i in range(maxit):
    #定义函数空间
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    tmr.send(f'第{i}次空间时间')

    u0 = space.function()
    du = space.function()
    diffusion_coef.uh = u0
    reaction_coef.uh = u0
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)

    #添加积分子
    D = ScalarDiffusionIntegrator(diffusion_coef, q=p+2)
    M = ScalarNonlinearMassIntegrator(reaction_coef, q=p+2)
    f = ScalarSourceIntegrator(pde.source, q=p+2)

    sform = NonlinearForm(space)
    sform.add_integrator([D, M])
    sform.add_integrator(f)
    bc = DirichletBC(space, gd=0.0, threshold=isDDof)

    while True:
        #矩阵组装、边界条件处理
        A, F = sform.assembly()
        tmr.send(f'第{i}次矩组装时间')
        A, F = bc.apply(A, F)

        #求解增量
        du = cg(A, F)
        u0 += du
        tmr.send(f'第{i}次求解器时间')

        #清除半线性积分子缓存
        M.clear()

        #计算误差
        err = bm.max(bm.abs(du))
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
print(NDof)

bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
ps = mesh.bc_to_point(bc)
u = pde.solution(ps)
uh = u0(bc)

next(tmr)
print(errorMatrix)
print(errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0], cellcolor=u, linewidths=0)
mesh.add_plot(axes[1], cellcolor=uh, linewidths=0) 
plt.show()
