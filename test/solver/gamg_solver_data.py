import numpy as np
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator,LinearForm,DirichletBC
    )
from fealpy.pde.poisson_2d import CosCosData
from fealpy.decorator import barycentric
from fealpy.fem import PoissonLFEMSolver
from fealpy.sparse import CSRTensor
from fealpy.utils import timer
from fealpy import logger
from fealpy.solver import GAMGSolver
logger.setLevel('INFO')

tmr = timer()
next(tmr)
pde = CosCosData()
nx = 10
ny = 10
m = 5
p=2
mesh = TriangleMesh.from_box(box = pde.domain(),nx=nx,ny=ny)
mesh.uniform_refine(n=5,returnim=True)
s0 = PoissonLFEMSolver(pde, mesh, p,timer=tmr,logger=logger)

init_data = [{'theta': 0.025, # 粗化系数
              'csize': 50, # 最粗问题规模
              'ctype': 'C', # 粗化方法
              'itype': 'T', # 插值方法
              'ptype': 'V', # 预条件类型
              'sstep':  2, # 默认光滑步数
              'isolver': 'MG', # 默认迭代解法器
              'maxit':  200,   # 默认迭代最大次数
              'csolver': 'direct', # 默认粗网格解法器
              'rtol': 1e-8,      # 相对误差收敛阈值
              'atol': 1e-8,      # 绝对误差收敛阈值
             }]
test_data = [{'A': s0.A ,
              'p': p,#基函数空间维数
              'm': 5,#加密次数
              'f' : s0.b,#最初右端项，用来检验残差,
              'nx':nx,
              'ny':ny,
              'domain':pde.domain(),
}]