import numpy as np
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorDiffusionIntegrator

@dataclass
class HarmonicMapData:
    """
    @brief Harmonic map data structure
    @param mesh: 计算网格
    @param didx: 狄利克雷点索引
    @param dval: 狄利克雷点值
    """

    mesh: TriangleMesh
    didx: np.ndarray
    dval: np.ndarray
    


def sphere_harmonic_map(data : HarmonicMapData):
    """
    @brief 计算平面到球面的调和映射
    """
    # 0. 数据准备
    mesh = data.mesh
    didx = data.didx
    dval = data.dval

    NN = mesh.number_of_nodes()
    GD = mesh.geo_dimension()
    gdof = NN*GD 

    space = LagrangeFiniteElementSpace(mesh, p=1)
    space.doforder = 'vdims'

    bform = BilinearForm((space, )*GD)
    bform.add_domain_integrator(VectorDiffusionIntegrator())
    S = bform.assembly() # 刚度矩阵

    # 1. 计算初值
    ## 1.1 狄利克雷边界条件处理
    idof = np.ones(gdof, dtype=np.bool_)
    for i in range(didx.size):
        idof[didx*GD+i] = 0
    T0 = spdiags(idof, 0, gdof, gdof)
    T1 = spdiags(1-idof, 0, gdof, gdof)
    S = T@S@T + T1
    ## 1.2 右端处理
    f = np.zeros(gdof, dtype=np.float64)
    f[~idof] = dval.fllaten
    f[idof] = -S@f

    ## 1.3 解方程
    uh = spsolve(S, f)

    # 2. 迭代求解
    I = 
    while True:
        ## 2.1 计算 C
        I = np.tile(np.arange(NN), (GD, 1)).T.flatten()
        J = np.arange(gdof)
        uh[~idof] = 0
        C = csr_matrix((uh, (I, J)), shape=(NN, gdof))






    





