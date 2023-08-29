# 导入相关模块
from fealpy.pde.truss_model import Truss_2d_four_bar

from fealpy.functionspace import LagrangeFESpace

from fealpy.fem import BilinearForm
from fealpy.fem import TrussStructureIntegrator
from fealpy.fem import DirichletBC

import numpy as np

# 建立模型
pde = Truss_2d_four_bar()

# 建立网格
mesh = pde.init_mesh()

# 建立连续的 Lagrange 有限元空间，一个节点的自由度全排完再排下一个
space = LagrangeFESpace(mesh, p=1, spacetype='C', doforder='vdims')

GD = mesh.geo_dimension()
vspace = GD*(space, ) # 把标量空间张成向量空间
uh = vspace[0].function(dim=GD) 

# 桁架所需的参数
E0 = pde.E # 杨氏模量 newton/mm^2
A0 = pde.A # 横截面积 mm^2

# 双线性型
bform = BilinearForm(vspace)
bform.add_domain_integrator(TrussStructureIntegrator(E0, A0))
K = bform.assembly() 

# 荷载初始化 
F = np.zeros((uh.shape[0], GD), dtype=np.float64)
# 施加节点力的索引和大小 
idx, f = mesh.meshdata['force_bc']
F[idx] = f 

# 位移边界条件
idx, disp = mesh.meshdata['disp_bc']
bc = DirichletBC(vspace, disp, threshold=idx)

# 求解
A, F = bc.apply(K, F.flat, uh)
