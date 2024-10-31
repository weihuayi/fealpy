import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags

from fealpy.functionspace import LagrangeFESpace as Space

from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import VectorMassIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.geometry.domain_2d import RectangleDomain

class BoxDomainData2d():
    """
    @brief 混合边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    def __init__(self, E=1.0, nu =0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.E=E
        self.nu=nu
        
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu=self.E/(2*(1+self.nu))
        
    def domain(self):
        return [0,1,0,1]
        
    def init_mesh(self, n = 1):
        """
        @brief 初始化网格
        @param[in] n 网格加密的次数，默认值为 1
        @return 返回一个初始化后的网格对象
        """
        h = 0.5
        domain = RectangleDomain()
        mesh = TriangleMesh.from_domain_distmesh(domain, h, output=False)
        mesh.uniform_refine(n)

        return mesh 

    def triangle_mesh(self):
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=10, ny=10)

        return mesh

    @cartesian
    def source(self, p):
        """
        @brief 返回给定点的源项值 f
        @param[in] p 一个表示空间点坐标的数组
        @return 返回源项值
        """
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0

        return val

    @cartesian
    def dirichlet(self, p):
        """
        @brief 返回 Dirichlet 边界上的给定点的位移
        @param[in] p 一个表示空间点坐标的数组
        @return 返回位移值，这里返回常数向量 [0.0, 0.0]
        """
        val = np.zeros((p.shape[0], 2), dtype=np.float64)
        # val = np.array([0.0, 0.0], dtype=np.float64)

        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定点是否在 Dirichlet 边界上
        @param[in] p 一个表示空间点坐标的数组
        @return 如果在 Dirichlet 边界上，返回 True，否则返回 False
        """
        x = p[..., 0]
        y = p[..., 1]
        flag1 = np.abs(x) < 1e-13
        flag2 = np.abs(x - 1) < 1e-13
        flagx = np.logical_or(flag1, flag2)
        flag3 = np.abs(y) < 1e-13
        flag4 = np.abs(y - 1) < 1e-13
        flagy = np.logical_or(flag3, flag4)
        flag = np.logical_or(flagx, flagy)

        return flag
    
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=2, type=int,
        help='Lagrange 有限元空间的次数, 默认为 2 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=5, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--scale',
        default=1, type=float,
        help='网格变形系数，默认为 1')

parser.add_argument('--doforder',
        default='vdims', type=str,
        help='自由度排序的约定，默认为 vdims')

args = parser.parse_args()
p = args.degree
GD = args.GD
n = args.nrefine
scale = args.scale
doforder = args.doforder

pde = BoxDomainData2d()
mu = pde.mu
lambda_ = pde.lam
domain = pde.domain()
mesh = pde.triangle_mesh()

space = Space(mesh, p=p, doforder=doforder)
uh = space.function(dim=GD)
vspace = GD*(space, )
gdof = vspace[0].number_of_global_dofs()
vgdof = gdof * GD
ldof = vspace[0].number_of_local_dofs()
vldof = ldof * GD

integrator1 = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=p+1)
bform = BilinearForm(vspace)
bform.add_domain_integrator(integrator1)
KK = integrator1.assembly_cell_matrix(space=vspace)
bform.assembly()
K = bform.get_matrix()

integrator2 = VectorMassIntegrator(c=1, q=5)
bform2 = BilinearForm(vspace)
bform2.add_domain_integrator(integrator2)
MK = integrator2.assembly_cell_matrix(space=vspace)
bform2.assembly()
M = bform2.get_matrix()

integrator3 = VectorSourceIntegrator(f = pde.source, q=5)
lform = LinearForm(vspace)
lform.add_domain_integrator(integrator3)
FK = integrator3.assembly_cell_vector(space = vspace)
lform.assembly()
F = lform.get_vector()

ipoints = space.interpolation_points()
fh = pde.source(p=ipoints)
fh_1 = np.zeros(M.shape[0])
fh_1[::GD] = fh[:,0]
fh_1[1::GD] = fh[:,1]
Fh = M @ fh_1

if hasattr(pde, 'dirichlet'):
    # dflag.shape = (gdof, GD)
    dflag = vspace[0].boundary_interpolate(gD=pde.dirichlet, uh=uh,
                                           threshold=pde.is_dirichlet_boundary)
    Fh -= K@uh.flat

    bdIdx = np.zeros(K.shape[0], dtype=np.int_)
    bdIdx[dflag.flat] = 1
    D0 = spdiags(1-bdIdx, 0, K.shape[0], K.shape[0])
    D1 = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
    K = D0@K@D0 + D1

    Fh[dflag.flat] = uh.ravel()[dflag.flat]
    #bc = DirichletBC(space=vspace, gD=pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    #K, Fh = bc.apply(K, Fh, uh)

uh.flat[:] = spsolve(K, Fh)
print("uh:\n", uh.shape, uh)

u_exact = space.interpolate(pde.solution)
print("u_exact:", u_exact.shape, "\n", u_exact)

output = './mesh_linear/'
if not os.path.exists(output):
    os.makedirs(output)
fname = os.path.join(output, 'linear_elastic.vtu')

mesh.nodedata['u'] = uh[:, 0]
mesh.nodedata['v'] = uh[:, 1]
mesh.to_vtk(fname=fname)

u_exact = space.interpolate(pde.solution)
print("u_exact:", u_exact.shape, "\n", u_exact)