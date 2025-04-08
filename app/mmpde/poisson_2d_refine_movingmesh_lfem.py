import argparse
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.tools.show import showmultirate, show_error_table

from fealpy.backend import backend_manager as bm
from fealpy.pde.poisson_2d import LShapeRSinData
from fealpy.mesh import TriangleMesh
from fealpy.old.mesh import TriangleMesh as TM
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.functionspace import LagrangeFESpace,ParametricLagrangeFESpace
from fealpy.fem import (BilinearForm,
                        ScalarDiffusionIntegrator,
                        LinearForm,
                        ScalarSourceIntegrator,
                        DirichletBC)
from app.mmpde.harmap_mmpde import *
from fealpy.solver import cg, spsolve
from scipy.sparse import spdiags

# 参数解析
parser = argparse.ArgumentParser(description="""高阶移动网格方法""")
parser.add_argument('--mdegree', 
       default=2, type=int,
       help='网格的阶数，默认为 2 次.')

parser.add_argument('--sdegree', 
       default=2, type=int,
       help='ParametricLagrangeFESpace空间的次数，默认为 2 次.')

parser.add_argument('--itype',
        default='iso', type=str,
        help='初始网格，默认直角三角形网格.')

parser.add_argument('--n',
        default=2, type=int,
        help='初始网格剖分规格.')

parser.add_argument('--h',
        default=1.7, type=bm.float64,
        help='初始网格剖分大小.')

parser.add_argument('--alpha',
        default=0.25, type=bm.float64,
        help='步长控制参数.')

parser.add_argument('--beta',
        default=0.15, type=bm.float64,
        help='控制函数参数.')

parser.add_argument('--moltimes',
        default=25, type=int,
        help='磨光次数.')

parser.add_argument('--mtype',
        default='ltri', type=str,
        help='网格类型， 默认三角形网格.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='网格加密次数.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy.")

args = parser.parse_args()
bm.set_backend(args.backend)

mdegree = args.mdegree
sdegree = args.sdegree
itype = args.itype
n = args.n
h = args.h
alpha = args.alpha
beta = args.beta
moltimes = args.moltimes
maxit = args.maxit
mtype = args.mtype

pde = LShapeRSinData()

if itype == 'iso':
    def thr(p):
        x = p[...,0]
        y=  p[...,1]
        area = bm.array([0.01,1,-1,-0.01,-0.4,0.4])
        in_x = (x >= area[0]) & (x <= area[1])
        in_y = (y >= area[2]) & (y <= area[3])
        if p.shape[-1] == 3:
            z = p[...,2]
            in_z = (z >= area[4]) & (z <= area[5])
            return in_x & in_y & in_z
        return  in_x & in_y

    tmesh0 = TriangleMesh.from_box([-1, 1, -1, 1], nx=n, ny=n, threshold=thr)
    tmesh1 = TriangleMesh.from_box([-1, 1, -1, 1], nx=n, ny=n, threshold=thr)
    mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh0, p=mdegree)
    fix_mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh1, p=mdegree)
elif itype == 'equ':
    vertices = bm.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],
                       [-1.0,1.0],[-1.0,-1.0],[0.0,-1.0]],dtype=bm.float64)
    mesh0 = TM.from_polygon_gmsh(vertices, h=h) 
    node = mesh0.entity('node')
    cell = mesh0.entity('cell')
    tmesh0 = TriangleMesh(node, cell)
    tmesh1 = TriangleMesh(node, cell)
    mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh0, p=mdegree)
    fix_mesh = LagrangeTriangleMesh.from_triangle_mesh(tmesh1, p=mdegree)
    
pro_mesh = mesh
NDof = bm.zeros(maxit, dtype=bm.int32)

# 可视化
def high_order_meshploter(mesh, uh= None, model='mesh', scat_node=True , scat_index = slice(None)):
    nodes = mesh.node
    n = mesh.p
    def lagrange_interpolation(points, num_points=100, n = n):
        """
        @brief 利用拉格朗日插值构造曲线
        @param points: 插值点的列表 [(x0, y0), (x1, y1), ..., (xp, yp)]
        @param num_points: 曲线上点的数量
        @param n: 插值多项式的次数
        @return: 曲线上点的坐标数组
        """
        t = bm.linspace(0, n, num_points)

        def L(k, t):
            Lk = bm.ones_like(t)
            for i in range(n + 1):
                if i != k:
                    Lk *= (t - i) / (k - i)
            return Lk
        GD = points.shape[-1]
        curve = bm.zeros((t.shape[0], points.shape[0],GD), dtype=bm.float64)
        for k in range(n + 1):
            Lk = L(k, t)
            for i in range(GD):
                xk = points[:,k,i]
                bm.add_at(curve , (...,i) ,Lk[:,None] * xk)
        return curve
    fig = plt.figure()
    edges = mesh.edge
    if model == 'mesh':
        p = nodes[edges]
        curve = lagrange_interpolation(p,n=n)
        plt.plot(curve[...,0], curve[...,1], 'b-',linewidth=0.5)
        if scat_node:
            plt.scatter(nodes[scat_index, 0], 
                        nodes[scat_index, 1],s = 3, color='r')  # 绘制节点
        plt.gca().set_aspect('equal')
        plt.axis('off') # 关闭坐标轴
    elif model == 'surface':
        points = bm.concat([nodes, uh[...,None]], axis=-1)
        ax = fig.add_subplot(111, projection='3d')
        p = points[edges]
        curve = lagrange_interpolation(p,n=n).transpose(1,0,2)
        nan_separator = bm.full((curve.shape[0], 1, 3), bm.nan)  # 每条曲线之间插入 NaN
        concatenated = bm.concat([curve, nan_separator], axis=1)  # 插入分隔符
        curve = concatenated.reshape(-1, 3) 
        ax.plot(curve[..., 0], curve[..., 1], curve[..., 2], linewidth=1)
        if scat_node:
            ax.scatter(points[scat_index, 0], 
                       points[scat_index, 1], 
                       points[scat_index, 2],s = 2, color='r')  # 绘制节点
    plt.show()
# poisson 求解
def poisson_solver(pde, mesh, p):
    space = ParametricLagrangeFESpace(mesh, p=p)
    NDof = space.number_of_global_dofs()
    print('NDof:', NDof)
    uh = space.function()
    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(method='isopara'))
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(source=pde.source, method='isopara'))
    A = bform.assembly()
    b = lform.assembly()
    A, b = DirichletBC(space, gd=pde.solution).apply(A, b)
    uh[:] = cg(A , b,maxiter=5000, atol=1e-14, rtol=1e-14)
    return uh

# 插值误差计算
def interplote_error(pde, mesh, p):
    space = ParametricLagrangeFESpace(mesh=mesh, p=p)
    uI = space.interpolate(pde.solution)
    error = mesh.error(pde.solution , uI)
    return error


# 移动网格前的求解
uh = poisson_solver(pde=pde, mesh=fix_mesh, p=sdegree)

# 网格移动
Vertex = bm.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],
                   [-1.0,1.0],[-1.0,-1.0],[0.0,-1.0]],dtype=bm.float64)

error_matrix_0 = bm.zeros(maxit+1)
error_matrix_1 = bm.zeros(maxit+1) 
error_matrix_0[0] = interplote_error(pde=pde , mesh=pro_mesh , p=sdegree)
error_matrix_1[0] = interplote_error(pde=pde, mesh=fix_mesh , p=sdegree)

# 移动网格加密
for i in range(maxit):
    MDH = Mesh_Data_Harmap(pro_mesh, Vertex)
    Vertex_idx , Bdinnernode_idx ,sort_BdNode_idx= MDH.get_basic_infom()

    HMP = Harmap_MMPDE(pro_mesh, uh, beta, Vertex_idx, 
                    Bdinnernode_idx, sort_BdNode_idx=sort_BdNode_idx, alpha = alpha,
                    mol_times=moltimes)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    M1 = HMP.M
    M = 1/M1
    high_order_meshploter(pro_mesh, M, 'surface')

    if i == 3:
        alpha = 0.125
        HMP.tol = 3.5e-2
    pro_mesh, uh = HMP.mesh_redistribution(uh, pde=pde)
    high_order_meshploter(pro_mesh, model="mesh")
    pro_mesh.uniform_refine()
    high_order_meshploter(pro_mesh, model="mesh")
    fix_mesh.uniform_refine()
    error0 = interplote_error(pde=pde, mesh=pro_mesh, p=sdegree)
    error1 = interplote_error(pde=pde, mesh=fix_mesh, p=sdegree)
    error_matrix_0[i+1] = error0
    error_matrix_1[i+1] = error1
    
    uh = poisson_solver(pde, pro_mesh, p=sdegree)

print(error_matrix_0)
print(error_matrix_1)

pro_order = bm.log2(error_matrix_0[:-1]/error_matrix_0[1:])
fix_order = bm.log2(error_matrix_1[:-1]/error_matrix_1[1:])
print('pro_order:', pro_order)
print('fix_order:', fix_order)

# 绘制误差折线图
plt.figure()
plt.plot(range(1,maxit+1), error_matrix_0[1:], 'o-', label='moved_mesh_refined')
plt.plot(range(1,maxit+1), error_matrix_1[1:], '^-', label='uniform_mesh_refined')
plt.xlabel('Refinement step')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error')
plt.legend()

# 绘制收敛阶折线图
plt.figure()
plt.plot(range(1, maxit), pro_order[1:], 'o-', label='moved_mesh_refined')
plt.plot(range(1, maxit), fix_order[1:], '^-', label='uniform_mesh_refined')
plt.xlabel('Refinement step')
plt.ylabel('Convergence rate')
plt.ylim(1, max(fix_order[1:].max(), pro_order[1:].max()) + 1)  # 设置 y 轴从 1 开始
plt.title('Convergence rate')
plt.legend()
plt.show()
