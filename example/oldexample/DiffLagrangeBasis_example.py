
import numpy as np
from itertools import combinations
from fealpy.mesh import MeshFactory as MF



def grad_basis(bc, p):

    TD = bc.shape[-1] - 1
    c = np.arange(1, p+1, dtype=self.itype)
    P = 1.0/np.multiply.accumulate(c)

    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = np.ones(shape, dtype=np.float64)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
    B = np.cumprod(A, axis=-2) # 函数值
    C = np.zeros(shape, dtype=np.float64) # 一阶导数
    D = np.zeros(shape, dtype=np.float64) # 二阶导数
    D = np.zeros(shape, dtype=np.float64) # 二阶导数

p = 6
mesh = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri')

a = np.arange(1, 3)





