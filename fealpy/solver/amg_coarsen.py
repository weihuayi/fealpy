
from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags
from fealpy.sparse import csr_matrix
import time

def ruge_stuben_coarsen(A, theta=0.025):
    
    """Ruge-Stuben coarsening method for multigrid preconditioning.
    
    This method applies the Ruge-Stuben coarsening technique for multigrid methods.
    It constructs the coarse grid operator by selecting a set of coarse nodes based 
    on the given matrix A. The method reduces the problem size by creating a set of 
    coarse variables (the C-set) and uses a interpolation operator (Pro) to map from 
    the fine grid to the coarse grid.

    Parameters:
        A (CSRMatrix): The input matrix representing the linear system. It should be a sparse matrix.
        theta (float, optional): A threshold parameter used to delete weak connections in the matrix. Default is 0.025.

    Returns:
        Pro (CSRMatrix): The interpolation operator that maps from the fine grid to the coarse grid.
        Res (CSRMatrix): The restriction operator that maps from the coarse grid to the fine grid.

        
    Notes:
        The method assumes that `A` is a M sparse matrix and uses it to construct a coarse grid.
        The interpolation and restriction operators are constructed using the Ruge-Stuben approach.
    """
    
    N = A.shape[0]
    maxaij = A.col_min()+0.05
    inverse_maxaij = 1 / bm.abs(maxaij)
    D = spdiags(inverse_maxaij,diags=0,M = N,N =N)
    Am = D @ A

    # Delete weak connectness
    im, jm, sm = Am.find()
    idx = (-sm > theta)
    As = csr_matrix((bm.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
    Am = csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    Ass = (As + As.T) / 2.0

    isF = bm.zeros(N, dtype=bool)
    degIn = bm.array(As.sum(axis=1)).flatten()
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
        
        i, j = Ass[S,S].triu(k=1).nonzero_slice
        
        idx = degInAll[S[i]] >= degInAll[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        C = bm.where(isC)[0]
        i ,_= Ass[:, C].nonzero_slice
        
        
        isF[i] = True
        U = bm.where(~(isF | isC))[0]

        degIn[isF | isC] = 0
        degFin = bm.zeros(N)
        
        F = bm.where(isF)[0]
        if U.shape[0] == 0:
            degFin = degFin
        else:
            degFin[U] = (As[F, U]).sum(axis=1)
        
        if len(U) <= 20:
            isC[U] = True
            U = []
            
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

    Afc = Am[fineNode, coarse2fine]

    Dsum = 1 / bm.array(Afc.sum(axis=0)+0.1).flatten()
    k = Dsum.shape[0]
    indptr = bm.arange(k)
    indices = bm.arange(k)
    Dsum = csr_matrix((Dsum, (indices, indptr)), shape=(k, k))
    
    ti, tj= (Dsum @ Afc).nonzero_slice
    tw = (Dsum @ Afc).data

    ip = bm.concatenate((ip, fineNode[ti]))
    jp = bm.concatenate((jp, tj))
    sp_vals = bm.concatenate((sp_vals, tw))
    Pro = csr_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.T

    return Pro, Res

def ruge_stuben_chen_coarsen(A, theta=0.025):
    """
    @brief Long Chen 修改过的 Ruge-Stuben 粗化方法

    @param[in] A 对称正定矩阵
    @param[in] theta 粗化阈值
    """

    # 1. 初始化参数
    N = A.shape[0]
    isC = bm.zeros(N, dtype=bm.bool)
    N0 = min(int(bm.floor(bm.sqrt(N))), 25)

    # 2. 生成强连通矩阵 
    # 然后函数计算出归一化的矩阵Am（矩阵A的对角线被归一化），
    # 并找出强连接的节点，也就是那些Am的元素值小于阈值theta的节点。
    # 得到的结果保存在矩阵G中。
    Dinv = spdiags(1./bm.sqrt(A.diags().values),diags=0,M = N,N =N)
    
    Am = Dinv @ A @ Dinv # 对角线归一化矩阵
    im, jm, sm = Am.find()
    flag = (-sm > theta) 
    # 删除对角、非对角弱联接项，注意对角线元素为1，也会被过滤掉
    G = csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))

    # 3. 计算顶点的度 
    # 函数计算出每个节点的度，也就是与每个节点强连接的节点数量。
    # 如果有太多的节点没有连接，函数会随机选择N0个节点作为粗糙节点并返回。
    deg = bm.tensor(G.astype(bm.bool).sum(axis=0).flat,dtype=bm.float64)
    # deg = bm.tensor(bm.sum(csr_matrix(G, dtype=bm.bool), axis=1).flat,
    #         dtype=bm.float64)
    
    if bm.sum(deg > 0) < 0.25*bm.sqrt(N):
        isC[bm.random.choice(range(N), N0)] = True
        return isC, G

    flag = (deg > 0)
    deg[flag] += 0.1 * bm.random.rand(bm.sum(flag))

    # 4. 寻找最大独立集 
    # 函数尝试找出一个近似的最大独立集并将其节点添加到粗糙节点集合中。
    # 如果某节点被标记为粗糙节点，则其相邻的节点会被标记为细节点。
    isF = bm.zeros(N, dtype=bm.bool)
    isF[deg == 0] = True # 孤立点为细节点 
    isU = bm.ones(N, dtype=bm.bool) # 未决定的集合

    while bm.sum(isC) < N/2 and bm.sum(isU) > N0:
        # 如果粗节点的个数少于总节点个数的一半，并且未决定的点集大于 N0
        isS = bm.zeros(N, dtype=bm.bool) # 选择集
        isS[deg>0] = True # 从非孤立点选择
        S = bm.nonzero(isS)[0]
        # 非孤立点集的连接关系
        #i, j = sp.triu(G[S, :][:, S], 1).nonzero()
        i, j = G[S, S].triu(1).nonzero_slice
        # 第 i 个非孤立点的度大于等于第 j 个非孤立点的度
        flag = deg[S[i]] >= deg[S[j]]
        isS[S[j[flag]]] = False # 把度小的节点从选择集移除
        isS[S[i[~flag]]] = False # 把度小的节点从选择集移除
        isC[isS] = True # 剩下的点就是粗点
        C = bm.nonzero(isC)[0]
        # Remove coarse nodes and neighboring nodes from undecided set
        i, _, _ = G[:, C].find()
        isF[i] = True # 粗点的相邻点是细点
        isU = ~(isF | isC) # 不是细点也不是粗点，就是未决定点
        deg[~isU] = 0 # 粗点或细节的度设置为 0

        if bm.sum(isU) <= N0:
            # 如果未决定点的数量小于等于 N0，把未决定点设为粗点
            isC[isU] = True
            isU = []

    return isC, G

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
    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    NC = N - len(fineNode)
    coarseNode = bm.arange(NC)
    coarseNodeFineIdx = bm.nonzero(isC)[0]

    # 2. 构造延长和限止算子
    Acf = A[coarseNodeFineIdx, fineNode] # 获取粗-->细矩阵块拿出来
    Dsum = bm.asarray(Acf.sum(axis=0)).reshape(-1) # 每个细点对应粗点值的和
    flag = (Dsum != 0) # 和非零的细节点标记数组
    NF = bm.sum(flag) # 细节点的个数
    Dsum = spdiags(1./Dsum[flag],diags=0,M = NF,N =NF) # 形成一个稀疏对角矩阵
    #Dsum = bm.diags(1./Dsum[flag], 0) # 形成一个稀疏对角矩阵
    flag = bm.nonzero(flag)[0]
    i, j, w = (Acf[:, flag] @ Dsum).find() # 每一列都除以这一列的和
    # 注意 tj 代表细节点的编号， ti 代表粗节点的编号
    # 而延长矩阵负责把粗网格的信息变到细网格上
    I = bm.concatenate((coarseNodeFineIdx, fineNode[j]))
    J = bm.concatenate((coarseNode, i))
    val = bm.concatenate((bm.ones(NC), w))
    P = csr_matrix((val, (I, J)), shape=(N, NC))
    R = P.T
    return P, R

def aggregation_coarsen( A, theta=0.025):
    """
    @brief 
    """
    N = A.shape[0]
    isC = bm.zeros(N, dtype=bool)
    N0 = min(int(bm.sqrt(N)), 25)

    # Initialize output
    node2agg = bm.zeros(N, dtype=int)
    agg2node = bm.zeros(N, dtype=int)

    # Generate strong connectness matrix
    Dinv = spdiags(1./bm.sqrt(A.diags().values),diags=0,M = N,N =N)
    Am = Dinv @ A @ Dinv
    im, jm, sm = Am.find()
    idx = (-sm > theta)
    As = csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    As += spdiags(bm.ones(N),diags=0,M = N,N =N)
    As1 = As.astype(bm.bool)
    As2 = (As1 @ As1).triu(1)

    # Compute degree of vertex
    deg = As1.sum(axis=0)
    deg = bm.squeeze(bm.asarray(deg))
    if bm.sum(deg>0) < 0.25*bm.sqrt(N):
        isC[bm.random.choice(range(N), N0)] = True
        agg2node = bm.where(isC)[0]
        node2agg[isC] = bm.arange(len(agg2node))
        return node2agg, As

    idx = (deg>0)
    deg = deg.astype(float)
    deg[idx] += 0.1 * bm.random.rand(bm.sum(idx))

    # Find an approximate maximal independent set and put to C set
    isF = bm.zeros(N, dtype=bool)
    isU = bm.ones(N, dtype=bool)
    isS = bm.ones(N, dtype=bool)
    isF[deg == 0] = True
    aggN = 0
    while aggN < N/2 and bm.sum(isS) > N0:
        isS = bm.zeros(N, dtype=bool)
        isS[deg>0] = True
        S = bm.where(isS)[0]
        S_As2 = As2[S,S]
        i, j ,_= S_As2.find()
        idx = deg[S[i]] >= deg[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        # Add new agg
        newC = bm.where(isS)[0]
        newAgg = aggN + bm.arange(len(newC))
        aggN += len(newC)
        node2agg[newC] = newAgg
        agg2node[newAgg] = newC

        # Remove coarse nodes and add neighboring nodes to the aggregate
        U = bm.where(isU)[0]
        i, j ,_= As[U,newC].find()
        isF[U[i]] = True
        isU = ~(isF | isC)
        node2agg[U[i]] = node2agg[newC[j]]
        deg[newC] = 0
        deg[U[i]] = 0
        U = bm.where(isU)[0]
        F = bm.where(isF)[0]
        i,_, _ = As[U,F].find()
        deg[U[i]] = 0

    agg2node = agg2node[:max(node2agg)+1]

    # Add left vertices to existing agg
    while any(isU):
        U = bm.where(isU)[0]
        i, j,_ = As[:, U].find()
        neighborAgg = node2agg[i]
        idx = (neighborAgg > 0)
        nAgg, neighborAgg = bm.unique(neighborAgg[idx], return_counts=True)
        isbdU = (nAgg > 0)
        isbdU = bm.where(isbdU)[0]
        print(U.shape, isbdU.shape)
        bdU = U[isbdU]
        node2agg[bdU] = neighborAgg[isbdU]
        isF[bdU] = True
        isU[bdU] = False

    return node2agg, As
