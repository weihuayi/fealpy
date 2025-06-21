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
        三维光滑元有限元方法插值
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
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--device',
        default='cpu', type=str,
        help='默认gpu上运行')


args = parser.parse_args()
tmr = timer()
next(tmr)
bm.set_backend(args.backend)
#device = "cuda"
p = args.degree
m = args.m
n = args.n
maxit = args.maxit
device = args.device
import torch 
torch.set_printoptions(precision=10)
if args.backend=="pytorch": 
    bm.set_default_device(device)

x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
#u = sp.sin(2*x)*sp.sin(2*y)*sp.sin(z)
#u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*z))
#u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x)*sp.sin(sp.pi*z))**2
#u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x)*sp.sin(sp.pi*z))
#u = sp.sin(5*x)*sp.sin(5*y)*sp.sin(5*z)
#u = sp.sin(7*x)*sp.sin(7*y)*sp.sin(7*z)
u = sp.sin(6*x)*sp.sin(6*y)*sp.sin(6*z)
#u = (sp.sin(2*sp.pi*y)*sp.sin(2*sp.pi*x)*sp.sin(sp.pi*z))**2
flist = get_flist(u)
NDof = bm.zeros(maxit, dtype=bm.float64)

errorType = ['$|| u - u_h||_{\\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\\Omega,0}$',
             '$||\\nabla^2 u - \\nabla^2 u_h||_{\\Omega,0}$',
             '$||\\nabla^3 u - \\nabla^3 u_h||_{\\Omega,0}$']
errorMatrix = bm.zeros((4, maxit), dtype=bm.float64)
tmr.send('网格和pde生成时间')
nx = n
for i in range(maxit):
    mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], nx, nx, nx)
    node = mesh.entity('node')
    isCornerNode = bm.zeros(len(node),dtype=bm.bool)
    for n in bm.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1],[0,0,1],[1,0,1],[0,1,1]], dtype=bm.float64):
        isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)
    #mesh.node = node/2
    from fealpy.functionspace.cm_conforming_fe_space3d_old import CmConformingFESpace3d as CmConformingFESpace3d_old
    #space = CmConformingFESpace3d_old(mesh,9, 1, isCornerNode)
    space = CmConformingFESpace3d(mesh, p=p, m=m, isCornerNode=isCornerNode)
    gdof = space.number_of_global_dofs()
    print("gdof", gdof)

    tmr.send(f'第{i}次空间生成时间')
    fI = space.function()
    fI[:] = space.interpolation(flist)
    print('插值')
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
    errorMatrix[0, i] = mesh.error(flist[0], fI)
    print('error1')
    errorMatrix[1, i] = mesh.error(flist[1], ug1val)
    print('error2')
    errorMatrix[2, i] = mesh.error(flist[2], ug2val)
    print('error3')
    errorMatrix[3, i] = mesh.error(flist[3], ug3val)
    print('error4')
    print(errorMatrix)
    
    nx = nx*2
    #n = int(n)




next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("order : ", bm.log2(errorMatrix[1,:-1]/errorMatrix[1,1:]))
print("order : ", bm.log2(errorMatrix[2,:-1]/errorMatrix[2,1:]))
print("order : ", bm.log2(errorMatrix[3,:-1]/errorMatrix[3,1:]))







