import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.linear_elasticity_model import BoxDomainData2d, BoxDomainData3d
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace as Space

from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator #TODO

from timeit import default_timer as timer

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=2, type=int,
        help='初始网格加密的次数, 默认初始加密 2 次.')

parser.add_argument('--scale',
        default=1, type=float,
        help='网格变形系数，默认为 1')

parser.add_argument('--doforder',
        default='sdofs', type=str,
        help='自由度排序的约定，默认为 sdofs')

args = parser.parse_args()
p = args.degree
GD = args.GD
n = args.nrefine
scale = args.scale
doforder = args.doforder

if GD == 2:
    pde = BoxDomainData2d()
    mesh = pde.init_mesh(n=n)
elif GD == 3:
    pde = BoxDomainData3d()
    mesh = TetrahedronMesh.from_box()

domain = pde.domain()
NN = mesh.number_of_nodes()

# 新接口程序
# 构建双线性型，表示问题的微分形式
space = Space(mesh, p=p, doforder=doforder)
uh = space.function(dim=GD)
vspace = GD*(space, ) # 把标量空间张成向量空间
bform = BilinearForm(vspace)
bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
bform.assembly()

# 构建单线性型，表示问题的源项
lform = LinearForm(vspace)
lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
if hasattr(pde, 'neumann'):
    bi = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
    lform.add_boundary_integrator(bi)
lform.assembly()

A = bform.get_matrix()
F = lform.get_vector()

if hasattr(pde, 'dirichlet'):
    bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A, F, uh)

uh.flat[:] = spsolve(A, F)
mesh.nodedata['uh'] = uh
mesh.to_vtk(fname='linear_lfem.vtu')
