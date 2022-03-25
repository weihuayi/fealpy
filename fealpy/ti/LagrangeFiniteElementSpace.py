import taichi as ti
import numpy as np

from scipy.sparse import csr_matrix
from .core import multi_index_matrix

def multi_index_matrix0d(p):
    multiIndex = 1
    return multiIndex 

def multi_index_matrix1d(p):
    ldof = p+1
    multiIndex = np.zeros((ldof, 2), dtype=np.int_)
    multiIndex[:, 0] = np.arange(p, -1, -1)
    multiIndex[:, 1] = p - multiIndex[:, 0]
    return multiIndex

def multi_index_matrix2d(p):
    ldof = (p+1)*(p+2)//2
    idx = np.arange(0, ldof)
    idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
    multiIndex = np.zeros((ldof, 3), dtype=np.int_)
    multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
    multiIndex[:,1] = idx0 - multiIndex[:,2]
    multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
    return multiIndex

def multi_index_matrix3d(p):
    ldof = (p+1)*(p+2)*(p+3)//6
    idx = np.arange(1, ldof)
    idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
    idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
    idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
    idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
    multiIndex = np.zeros((ldof, 4), dtype=np.int_)
    multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
    multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
    multiIndex[1:, 1] = idx0 - idx2
    multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
    return multiIndex

multi_index_matrix = [multi_index_matrix0d, multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

@ti.data_oriented
class LagrangeFiniteElementSpace():
    """
    单纯型网格上的任意次拉格朗日空间，这里的单纯型网格是指
    * 区间网格(1d)
    * 三角形网格(2d)
    * 四面体网格(3d)
    """
    def __init__(self, mesh, p=1, spacetype='C', q=None):
        self.mesh = mesh

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        self.p = ti.field(self.itype, shape=())
        self.p[None] = p

        TD = self.mesh.top_dimension()
        mi = multi_index_matrix[TD](p)
        self.multiIndex = ti.field(self.itype, shape=mi.shape)
        self.multiIndex.from_numpy(mi)


    def geo_dimension(self):
        return self.mesh.node.shape[0]

    def top_dimension(self):
        return self.multiIndex.shape[1] - 1

    def number_of_local_dofs(self):
        return self.multiIndex.shape[0]

    @ti.func
    def lagrange_shape_function(self, bc: ti.template()) -> (ti.template(), ti.template()):

        m = self.p[None] + 1
        n = self.multiIndex.shape[1]
        R0 = ti.Matrix.one(self.ftype, m, n)
        R1 = ti.Matrix.one(self.ftype, m, n)
        for i in range(1, self.multiIndex.shape[0]):
            pass

        return R0, R1

    @ti.kernel
    def test(self):
        R0, R1 = self.lagrange_shape_function(bc)
        print(R0)
        print(R1)


