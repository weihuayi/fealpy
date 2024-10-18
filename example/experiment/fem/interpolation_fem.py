import argparse
import sympy as sp
from fealpy.backend import backend_manager as bm
from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d
from fealpy.pde.biharmonic_triharmonic_3d import DoubleLaplacePDE, get_flist
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.decorator import barycentric
from fealpy.utils import timer

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        光滑元有限元方法插值
        """)

parser.add_argument('--degree',
        default=11, type=int,
        help='光滑有限元空间的次数, 默认为 11 次.')

parser.add_argument('--m',
        default=1, type=int,
        help='C^m 光滑元')

parser.add_argument('--n',
        default=1, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=2, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='pytorch', type=str,
        help='默认后端为numpy')

args = parser.parse_args()
tmr = timer()
next(tmr)
bm.set_backend(args.backend)
decive = "cpu"
p = args.degree
m = args.m
n = args.n
maxit = args.maxit

bm.set_default_device('cpu')
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
#u = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*z))**4
flist = get_flist(u)
NDof = bm.zeros(maxit, dtype=bm.float64)

errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega,0}$',
             '$||\\nabla^2 u - \\nabla^2 u_h||_{\\Omega,0}$',
             '$||\\nabla^3 u - \\nabla^3 u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((4, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')

for i in range(maxit):
    mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], 2**i, 2**i, 2**i)
    node = mesh.entity('node')
    isCornerNode = bm.zeros(len(node),dtype=bm.bool)
    for n in bm.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1],[0,0,1],[1,0,1],[0,1,1]], dtype=bm.float64):
        isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)
    mesh.node = node/2
    space = CmConformingFESpace3d(mesh, p=p, m=m, isCornerNode=isCornerNode)
    tmr.send(f'第{i}次空间生成时间')
    fI = space.interpolation(flist)
    tmr.send(f'第{i}次插值时间')

    @barycentric
    def ug1val(p):
        return fI.grad_m_value(p, 1)

    @barycentric
    def ug2val(p):
        return fI.grad_m_value(p, 2)
    @barycentric
    def ug3val(p):
        return fI.grad_m_value(p, 3)

    errorMatrix[0, i] = mesh.error(flist[0], fI, q=p+3)
    import ipdb
    ipdb.set_trace()
    errorMatrix[1, i] = mesh.error(flist[1], ug1val, q=p+3)
    #errorMatrix[2, i] = mesh.error(flist[2], ug2val, q=p+3)
    #errorMatrix[3, i] = mesh.error(flist[3], ug3val, q=p+3)
    print(errorMatrix)


next(tmr)


next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("order : ", bm.log2(errorMatrix[1,:-1]/errorMatrix[1,1:]))
print("order : ", bm.log2(errorMatrix[2,:-1]/errorMatrix[2,1:]))







