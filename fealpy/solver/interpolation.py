import numpy as np
import scipy.sparse as sp

def standard_interpolation(A, isC):
    """
    @brief 生成延长和限止矩阵 

    @param[in] A 一个对称正定矩阵
    @param[in] isC 标记粗点的逻辑数组

    @note Prolongation 矩阵将粗网格上的解插值到四网格上; Restriction  矩阵将
    细网格上的残差限制到粗网格上。
    """

    N = A.shape[0]

    # 1. 索引映射：函数首先创建了一个从粗网格到细网格的索引映射。它找出了所
    #    有的粗节点和细节点，并存储了他们的索引。
    allNode = np.arange(N)
    fineNode = allNode[~isC]
    Nc = N - len(fineNode)
    coarseNode = np.arange(Nc)
    coarseNodeFineIdx = np.nonzero(isC)[0]

    # 2. 构造延长和限止算子
    Acf = A[coarseNodeFineIdx, :][:, fineNode] # 获取粗-->细矩阵块拿出来
    Dsum = np.asarray(Acf.sum(axis=0)).reshape(-1) # 每个细点对应粗点值的和
    flag = (Dsum != 0) # 和非零的细节点标记数组
    Nf = np.sum(flag) # 细节点的个数
    Dsum = sp.diags(1./Dsum[flag], 0) # 形成一个稀疏对角矩阵
    ti, tj, tw = sp.find(Acf[:, flag] @ Dsum) # 每一列都除以这一列的和
    # 注意 tj 代表细节点的编号， ti 代表粗节点的编号
    # 而延长矩阵负责把粗网格的信息变到细网格上
    I = np.concatenate((coarseNodeFineIdx, fineNode[tj]))
    J = np.concatenate((coarseNode, ti))
    val = np.concatenate((np.ones(Nc), tw))
    P = sp.csr_matrix((val, (I, J)), shape=(N, Nc))
    R = P.transpose()
    return P, R

def two_points_interpolation(A, isC):
    """
    """
    N = A.shape[0]

    # Index map between coarse grid and fine grid
    allNode = np.arange(N)
    fineNode = allNode[~isC]
    HB = np.zeros((len(fineNode), 3))
    HB[:,0] = fineNode
    Nc = N - len(fineNode)
    coarseNode = np.arange(Nc)
    coarseNodeFineIdx = np.where(isC)[0]

    # Construct matrix-dependent prolongation and restriction operator
    Afc = A[fineNode,:][:,coarseNodeFineIdx]
    i, j, s = sp.find(Afc)
    HB[i[::-1],2] = j[::-1]
    HB[i,1] = j
    w = np.zeros((len(i), 2))
    w[i[::-1],1] = s[::-1]
    w[i,0] = s
    Dsum = np.sum(w, axis=1)
    idx = Dsum != 0
    w = w[idx,:] / Dsum[idx,None]
    HB = HB[idx,:]
    ip = np.concatenate((coarseNodeFineIdx, HB[:,0], HB[:,0]))
    jp = np.concatenate((coarseNode, HB[:,1], HB[:,2]))
    sp_vals = np.concatenate((np.ones(Nc), w[:,0], w[:,1]))
    Pro = sp.csr_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.transpose()

    return Pro, Res

def smoothing_aggregation_interpolation(A, node2agg, omega=0.35, smoothingstep=2):
    """
    """
    N = A.shape[0]
    Nc = max(node2agg)
    idx = np.where(node2agg != 0)[0]
    Pro = sp.csr_matrix((np.ones(len(idx)), (idx, node2agg[idx]-1)), shape=(N, Nc))

    # Smooth the piecewise constant prolongation
    for k in range(smoothingstep):
        Pro = Pro - omega * (A @ Pro)

    # Normalize the prolongation such that the row sum is one
    rowsum = np.array(Pro.sum(axis=1)).flatten()
    D = sp.diags(1. / rowsum)
    Pro = D @ Pro
    Res = Pro.transpose()

    return Pro, Res

def interpolation_n(A, isC):
    """
    """
    N = A.shape[0]
    allNode = np.arange(N) + 1
    fineNode = allNode[~isC]
    Nf = len(fineNode)
    Nc = N - Nf
    coarseNode = np.arange(Nc) + 1
    coarse2fine = np.where(isC)[0] + 1
    ip = coarse2fine
    jp = coarseNode
    sp = np.ones(Nc)

    Afc = A[fineNode-1][:, coarse2fine-1]
    Dsum = sp.diags(A[fineNode-1, fineNode-1])
    alpha = (np.sum(A[fineNode-1, :], axis=1) - Dsum.diagonal()) / np.sum(Afc, axis=1)
    Dsum = sp.diags(-alpha / Dsum.diagonal())
    ti, tj, tw = sp.find(Dsum @ Afc)
    ip = np.concatenate([ip, fineNode[ti]])
    jp = np.concatenate([jp, tj + 1])
    sp = np.concatenate([sp, tw])
    Pro = sp.csr_matrix((sp, (ip-1, jp-1)), shape=(N, Nc))
    Res = Pro.transpose()

    return Pro, Res
