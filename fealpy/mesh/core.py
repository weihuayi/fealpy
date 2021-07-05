"""

Notes
-----
   网格模块的核心部分，涉及编号，形函数等内容。 

Authors
-------

Huayi Wei, weihuayi@xtu.edu.cn

"""
import numpy as np 

class LinearMeshDataStructure():

    def total_edge(self):
        NC = self.NC
        cell = self.cell
        localEdge = self.localEdge

        totalEdge = cell[:, localEdge].reshape(-1, 2)
        return totalEdge

    def total_face(self):
        NC = self.NC
        cell = self.cell
        localFace = self.localFace
        NVF = self.NVF

        totalFace = cell[:, localFace].reshape(-1, NVF)
        return totalFace

    def construct_face(self):
        """ 

        Notes
        -----
            构造面
        """
        NC = self.NC
        NFC = self.NFC
        NVF = self.NVF

        totalFace = self.total_face()
        index = np.sort(totalFace, axis=-1)
        I = index[:, 0]
        I += index[:, 1]*(index[:, 1] + 1)//2
        I += index[:, 2]*(index[:, 2] + 1)*(index[:, 2] + 2)//6
        if NVF == 4: 
            I += index[:, 3]*(index[:, 3] + 1)*(index[:, 3] + 2)*(index[:, 3] + 3)//24
        _, i0, j = np.unique(I, return_index=True, return_inverse=True)

        NF = i0.shape[0]
        self.NF = NF
        self.face = totalFace[i0, :]

        self.face2cell = np.zeros((NF, 4), dtype=self.itype)

        i1 = np.zeros(NF, dtype=self.itype)
        i1[j] = np.arange(NFC*NC, dtype=self.itype)

        self.face2cell[:, 0] = i0//NFC
        self.face2cell[:, 1] = i1//NFC
        self.face2cell[:, 2] = i0%NFC
        self.face2cell[:, 3] = i1%NFC

    def construct_edge(self, TD=2):
        """ 

        Notes
        -----
            TD == 2: 构造 edge 和 edge2cell
            TD == 3: 构造 edge
        """
        NC = self.NC
        NEC = self.NEC
        NVE = self.NVE

        totalEdge = self.total_edge()
        index = np.sort(totalEdge, axis=-1)
        I = index[:, 0] 
        I += index[:, 1]*(index[:, 1] + 1)//2

        _, i0, j = np.unique(I, return_index=True, return_inverse=True)
        NE = i0.shape[0]
        self.NE = NE
        self.edge = totalEdge[i0, :]
        self.cell2edge = np.reshape(j, (NC, NEC))

        if TD == 2:
            self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

            i1 = np.zeros(NE, dtype=self.itype)
            i1[j] = np.arange(NEC*NC, dtype=self.itype)

            self.edge2cell[:, 0] = i0//NEC
            self.edge2cell[:, 1] = i1//NEC
            self.edge2cell[:, 2] = i0%NEC
            self.edge2cell[:, 3] = i1%NEC


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

def lagrange_shape_function(bc, p, n=0):
    """

    Notes
    -----
    
    计算形状为 (..., TD+1) 的重心坐标数组 bc 中的每一个重心坐标处的 p 次
    Lagrange 形函数关于 TD+1 个重心坐标的 n 阶导数.
    
    注意当 n = 0 时, 返回的是函数值。
    """
    assert n <= p

    TD = bc.shape[-1] - 1
    multiIndex = multi_index_matrix[TD](p) 
    ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数 

    c = np.arange(1, p+1, dtype=np.int_)
    P = 1.0/np.multiply.accumulate(c)
    t = np.arange(0, p)
    shape = bc.shape[:-1]+(p+1, TD+1) # (NQ, p+1, TD+1)
    A = np.ones(shape, dtype=bc.dtype)
    A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
    np.cumprod(A, axis=-2, out=A)

    if n == 0:
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        R = np.prod(A[..., multiIndex, idx], axis=-1)
        return R 
    else:
        T = p*bc[..., None, :] - t.reshape(-1, 1) # (NQ, p, TD+1)
        F0 = A.copy() # (NQ, p+1, TD+1) 
        F1 = np.zeros(A.shape, dtype=bc.dtype)

        # (NQ, p, TD+1) = (NQ, p, TD+1)*(NQ, p, TD+1) + (NQ, p, TD+1)
        for i in range(1, n+1):
            for j in range(1, p+1):
                F1[..., j, :] = F1[..., j-1, :]*T[..., j-1, :] + i*p*F0[..., j-1, :]
            F0[:] = F1

        A[..., 1:, :] *= P.reshape(-1, 1)
        F0[..., 1:, :] *= P.reshape(-1, 1)
        
        Q = A[..., multiIndex, range(TD+1)]
        M = F0[..., multiIndex, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1) # (NQ, ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, TD+1)



def lagrange_grad_shape_function(bc, p): 
    """
    Notes
    -----
    
    计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次
    Lagrange 形函数值关于该重心坐标的梯度。
    """

    TD = bc.shape[-1] - 1
    multiIndex = multi_index_matrix[TD](p) 
    ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数

    c = np.arange(1, p+1)
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
