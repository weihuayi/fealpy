import numpy as np
import scipy.sparse as sp
from fealpy.backend import backend_manager as bm
from fealpy.sparse import COOTensor, CSRTensor

from fealpy.sparse import csr_matrix
#from scipy.sparse import csr_matrix

def generate_positive_definite_matrix(N, scale=2.0):
    """
    生成一个 N x N 的对称正定矩阵 A

    @param N: 矩阵的大小
    @param scale: 控制矩阵值的尺度因子
    @return: 对称正定矩阵 A
    """
    # 1. 生成随机矩阵
    A_rand = bm.random.randn(N, N)
    A = A_rand 
    # 2. 构造对称矩阵
    #A = (A_rand + A_rand.T) / 2  # 对称化
    
    # 3. 确保正定性
    # 给对角线加上足够大的值，确保正定
    A += bm.eye(N) * scale

    # 4. 获取非零元素的位置和数值
    rows, cols = bm.nonzero(A)  # 获取非零元素的行列索引
    data = A[rows, cols]        # 获取非零元素的数值

    # 5. 将这些元素转换为CSR矩阵格式
    A_sparse = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    return A_sparse


def ruge_stuben_coarsen(A, theta=0.025):
    """
    @brief Ruge-Stuben 粗化方法
    """
    N = A.shape[0]
    maxaij = A.to_dense().min(axis=0)
    inverse_maxaij = 1 / bm.abs(maxaij)


    # csr_matrix 的三个构建数组：indptr, indices, data
    indptr = bm.arange(N)
    indices = bm.arange(N)
    D = csr_matrix((inverse_maxaij, (indices, indptr)), shape=(N, N))
    #D = sp.diags(1/np.abs(maxaij).toarray().flatten())1
    Am = D @ A

    # Delete weak connectness
    im, jm, sm = Am.find()
    idx = (-sm > theta)
    As = csr_matrix((bm.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
    Am = csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    Ass = (As + As.T) / 2.0

    isF = bm.zeros(N, dtype=bool)
    degIn = bm.array(As.sum(axis=0)).flatten()
    isF[degIn == 0] = True

    # Find an approximate maximal independent set and put to C set
    isC = bm.zeros(N, dtype=bool)
    U = bm.arange(N)
    degFin = bm.zeros(N)
    while bm.sum(isC) < N / 2 and len(U) > 20:
        isS = bm.zeros(N, dtype=bool)
        degInAll = degIn + degFin
        isS[(bm.random.rand(N) < 0.85 * degInAll / bm.mean(degInAll)) & (degInAll > 0)] = True
        S = bm.where(isS)[0]

        i, j = bm.nonzero(bm.triu(Ass.to_dense()[S][:, S], 1))
        
        idx = degInAll[S[i]] >= degInAll[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        i, _ = bm.nonzero(Ass.to_dense()[:, isC])
        isF[i] = True
        U = bm.where(~(isF | isC))[0]

        degIn[isF | isC] = 0
        degFin = bm.zeros(N)
        print(As.shape)
        print(isF)
        print(U)
        #print(As.to_dense()[isF, U].shape)
        degFin[U] = bm.array((As.to_dense()[isF, :][:, U]).sum(axis=0)).flatten()

        if len(U) <= 20:
            isC[U] = True
            U = []

    print(f'Number of coarse nodes: {bm.sum(isC)}')

    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    Nf = len(fineNode)
    Nc = N - Nf
    coarseNode = bm.arange(Nc)
    coarse2fine = bm.where(isC)[0]
    fine2coarse = bm.zeros(N, dtype=int)
    fine2coarse[isC] = coarseNode
    ip = coarse2fine
    jp = coarseNode
    sp_vals = bm.ones(Nc)

    Afc = Am.to_dense()[fineNode, :][:, coarse2fine]
    
    Dsum = 1 / bm.array(Afc.sum(axis=1)).flatten()
    k = Dsum.shape[0]
    indptr = bm.arange(k)
    indices = bm.arange(k)
    Dsum = csr_matrix((Dsum, (indices, indptr)), shape=(k, k))
    
    ti, tj= bm.nonzero(Dsum @ Afc)
    tw = (Dsum @ Afc)[ti, tj]
    ip = bm.concatenate((ip, fineNode[ti]))
    jp = bm.concatenate((jp, tj))
    sp_vals = bm.concatenate((sp_vals, tw))
    Pro = csr_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.T

    Ac = Res @ A @ Pro
    return Ac, Pro, Res

def ruge_stuben_coarsen_1(A, theta=0.025):
    """
    @brief Ruge-Stuben 粗化方法
    """
    A = A.to_dense()    
    print(A)
    N = A.shape[0]
    maxaij = A.min(axis=0)
    print(maxaij)
    D = sp.diags(1/np.abs(maxaij).flatten())
    Am = D @ A
    print(Am)
    # Delete weak connectness
    im, jm, sm = sp.find(Am)
    idx = (-sm > theta)
    As = sp.coo_matrix((np.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
    Am = sp.coo_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    Ass = (As + As.transpose()) / 2.0

    isF = np.zeros(N, dtype=bool)
    degIn = np.array(As.sum(axis=0)).flatten()
    isF[degIn == 0] = True

    # Find an approximate maximal independent set and put to C set
    isC = np.zeros(N, dtype=bool)
    U = np.arange(N)
    degFin = np.zeros(N)
    while np.sum(isC) < N / 2 and len(U) > 20:
        isS = np.zeros(N, dtype=bool)
        degInAll = degIn + degFin
        isS[(np.random.rand(N) < 0.85 * degInAll / np.mean(degInAll)) & (degInAll > 0)] = True
        S = np.where(isS)[0]
    
        i = sp.find(sp.triu(Ass[S][:, S], 1))[0]
        j = sp.find(sp.triu(Ass[S][:, S], 1))[1]
        idx = degInAll[S[i]] >= degInAll[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        i= sp.find(Ass[:, isC])[0]
        isF[i] = True
        U = np.where(~(isF | isC))[0]

        degIn[isF | isC] = 0
        degFin = np.zeros(N)
        degFin[U] = np.array(As.todense()[isF, :][:, U].sum(axis=0)).flatten()

        if len(U) <= 20:
            isC[U] = True
            U = []

    print(f'Number of coarse nodes: {np.sum(isC)}')

    allNode = np.arange(N)
    fineNode = allNode[~isC]
    Nf = len(fineNode)
    Nc = N - Nf
    coarseNode = np.arange(Nc)
    coarse2fine = np.where(isC)[0]
    fine2coarse = np.zeros(N, dtype=int)
    fine2coarse[isC] = coarseNode
    ip = coarse2fine
    jp = coarseNode
    sp_vals = np.ones(Nc)

    Afc = Am.todense()[fineNode, :][:, coarse2fine]
    Dsum = sp.diags(1 / np.array(Afc.sum(axis=1)).flatten())
    ti, tj, tw = sp.find(Dsum @ Afc)
    ip = np.concatenate((ip, fineNode[ti]))
    jp = np.concatenate((jp, tj))
    sp_vals = np.concatenate((sp_vals, tw))
    Pro = sp.coo_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.transpose()

    Ac = Res @ A @ Pro
    return Ac, Pro, Res

N = 4
A = generate_positive_definite_matrix(N)
#Ac, Pro, Res = ruge_stuben_coarsen(A)
Ac1, Pro1, Res1 = ruge_stuben_coarsen_1(A)
# print(Pro1)
# print(Pro)

