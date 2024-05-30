import numpy as np
from scipy.sparse import csr_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass

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
    GD = dval.shape[1]
    gdof = NN*GD 

    space = LagrangeFiniteElementSpace(mesh, p=1)
    space.doforder = 'vdims'

    bform = BilinearForm((space, )*GD)
    bform.add_domain_integrator(VectorDiffusionIntegrator())
    S = bform.assembly() # 刚度矩阵

    # 1. 计算初值
    ## 1.1 狄利克雷边界条件处理
    idof = np.ones(gdof, dtype=np.bool_) # 内部自由度
    idof[didx[:, None] * GD + np.arange(GD)] = False

    N = gdof-len(didx)*GD # 内部自由度个数
    I = np.arange(N)
    d = np.ones(N, dtype=np.float64)
    T = csr_matrix((d, (I, np.where(idof)[0])), shape=(N, gdof))

    f = np.zeros(gdof, dtype=np.float64)
    f[~idof] = dval.flatten()
    f = (T@S@f)
    S = T@S@T.T

    ## 1.2 解方程
    uh = spsolve(S, f)

    ## 1.3 归一化
    vh = uh.reshape(-1, GD)
    uh = vh/np.linalg.norm(vh, axis=1, keepdims=True)
    uh = uh.reshape(-1)

    # 2. 迭代求解
    I = np.tile(np.arange(N//GD), (GD, 1)).T.flatten()
    J = np.arange(N)
    while True:
        ## 2.1 计算 C 并组装矩阵
        C = csr_matrix((uh, (I, J)), shape=(N//GD, N))
        A = bmat([[S, C.T], [C, None]], format='csr') 

        ## 2.2 计算右端 b
        b = np.zeros(N+N//GD, dtype=np.float64)
        b[:N] = -S@uh

        ## 2.3 解方程
        x = spsolve(A, b)
        wh = x[:N]

        ## 2.4 归一化
        vh = uh.reshape(-1, GD) + wh.reshape(-1, GD)
        uh1 = vh/np.linalg.norm(vh, axis=1, keepdims=True)
        print(np.linalg.norm(uh1.flat - uh))
        if np.linalg.norm(uh1.flat - uh) < 1e-8:
            uh = uh1.reshape(-1)
            break
        uh = uh1.reshape(-1)

    fun = np.zeros((NN*GD))
    fun[idof] = uh
    fun[~idof] = dval.flat
    return fun.reshape(-1, GD)









    





