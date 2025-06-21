
"""

Notes
-----
    生成多重指标矩阵

Authors
-------

Huayi Wei, weihuayi@xtu.edu.cn


"""
import numpy as np 
import operator as op


def multi_index_matrix0d(p):
    multiIndex = 1
    return multiIndex 

def multi_index_matrix1d(p):
    ldof = p+1
    multiIndex = np.zeros((ldof, 2), dtype=np.int)
    multiIndex[:, 0] = np.arange(p, -1, -1)
    multiIndex[:, 1] = p - multiIndex[:, 0]
    return multiIndex

def multi_index_matrix2d(p):
    ldof = (p+1)*(p+2)//2
    idx = np.arange(0, ldof)
    idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
    multiIndex = np.zeros((ldof, 3), dtype=np.int)
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
    multiIndex = np.zeros((ldof, 4), dtype=np.int)
    multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
    multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
    multiIndex[1:, 1] = idx0 - idx2
    multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
    return multiIndex

multi_index_matrix = [multi_index_matrix0d, multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]

def lagrange_shape_function(bc, p):
    """

    Notes
    -----
    
    计算形状为 (..., TD+1) 的重心坐标数组 bc 中的每一个重心坐标处的 p 次 Lagrange 形函数值。
    """

    TD = bc.shape[-1] - 1
    multiIndex = multi_index_matrix[TD](p) 
    ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数 

    c = np.arange(1, p+1, dtype=np.int)
    P = 1.0/np.multiply.accumulate(c)
    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = np.ones(shape, dtype=self.ftype)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
    np.cumprod(A, axis=-2, out=A)
    A[..., 1:, :] *= P.reshape(-1, 1)
    idx = np.arange(TD+1)
    phi = np.prod(A[..., multiIndex, idx], axis=-1)
    return phi[..., np.newaxis, :] # (..., 1, ldof)

def lagrange_grad_shape_function(bc, p): 
    """
    Notes
    -----
    
    计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次
    Lagrange 形函数值关于该重心坐标的梯度。
    """

    multiIndex = multi_index_matrix[TD](p) 
    ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数

    c = np.arange(1, p+1, dtype=self.itype)
    P = 1.0/np.multiply.accumulate(c)

    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1)
    A = np.ones(shape, dtype=bc.dtype)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

    FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
    FF[..., range(p), range(p)] = p
    np.cumprod(FF, axis=-2, out=FF)
    F = np.zeros(shape, dtype=bc.dtype)
    F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
    F[..., 1:, :] *= P.reshape(-1, 1)

    np.cumprod(A, axis=-2, out=A)
    A[..., 1:, :] *= P.reshape(-1, 1)

    Q = A[..., multiIndex, range(TD+1)]
    M = F[..., multiIndex, range(TD+1)]

    shape = bc.shape[:-1]+(ldof, TD+1)
    R = np.zeros(shape, dtype=bc.dtype)
    for i in range(TD+1):
        idx = list(range(TD+1))
        idx.remove(i)
        R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
    return R # (..., ldof, TD+1)

