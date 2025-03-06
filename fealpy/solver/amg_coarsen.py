
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
    Ass = As + Am

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
    return isC,Am


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

def ruge_stuben_chen_coarsen(A, theta=0.025):
    """
    @brief Modified Ruge-Stuben coarsening method by Long Chen

    @param[in] A A symmetric positive definite matrix
    @param[in] theta Coarsening threshold
    """

    # 1. Initialize parameters
    N = A.shape[0]
    isC = bm.zeros(N, dtype=bm.bool)
    N0 = min(int(bm.floor(bm.sqrt(N))), 25)

    # 2. Generate the strong connectivity matrix
    # The function first computes a normalized matrix Am (where the diagonal of A is normalized),
    # and identifies strongly connected nodes, i.e., nodes whose elements in Am are smaller than the threshold theta.
    # The resulting strong connection matrix is stored in G.
    Dinv = spdiags(1./bm.sqrt(A.diags().values), diags=0, M=N, N=N)

    Am = Dinv @ A @ Dinv  # Diagonal-normalized matrix
    im, jm, sm = Am.find()
    flag = (-sm > theta)  
    # Remove weak connections, including diagonal elements (diagonal elements are 1 and will be filtered out)
    G = csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))

    # 3. Compute vertex degree
    # The function calculates the degree of each node, which is the number of strongly connected nodes.
    # If too many nodes are unconnected, it randomly selects N0 nodes as coarse nodes and returns.
    deg = bm.tensor(G.astype(bm.bool).sum(axis=0).flat, dtype=bm.float64)

    if bm.sum(deg > 0) < 0.25 * bm.sqrt(N):
        isC[bm.random.choice(range(N), N0)] = True
        return isC, G

    flag = (deg > 0)
    deg[flag] += 0.1 * bm.random.rand(bm.sum(flag))

    # 4. Find the maximal independent set
    # The function attempts to find an approximate maximal independent set and adds its nodes to the coarse node set.
    # If a node is marked as a coarse node, its neighboring nodes are marked as fine nodes.
    isF = bm.zeros(N, dtype=bm.bool)
    isF[deg == 0] = True  # Isolated points are fine nodes
    isU = bm.ones(N, dtype=bm.bool)  # Undecided nodes

    while bm.sum(isC) < N/2 and bm.sum(isU) > N0:
        # If the number of coarse nodes is less than half of the total nodes and the number of undecided nodes is greater than N0
        isS = bm.zeros(N, dtype=bm.bool)  # Selection set
        isS[deg > 0] = True  # Select from non-isolated nodes
        S = bm.nonzero(isS)[0]
        # Connectivity of the non-isolated node set
        i, j = G[S, S].triu(1).nonzero_slice
        # Check if the degree of node i is greater than or equal to that of node j
        flag = deg[S[i]] >= deg[S[j]]
        isS[S[j[flag]]] = False  # Remove lower-degree nodes from the selection set
        isS[S[i[~flag]]] = False  # Remove lower-degree nodes from the selection set
        isC[isS] = True  # The remaining nodes are coarse nodes
        C = bm.nonzero(isC)[0]
        # Remove coarse nodes and their neighbors from the undecided set
        i, _, _ = G[:, C].find()
        isF[i] = True  # Neighboring nodes of coarse nodes are fine nodes
        isU = ~(isF | isC)  # Nodes that are neither fine nor coarse remain undecided
        deg[~isU] = 0  # Set the degree of coarse and fine nodes to 0

        if bm.sum(isU) <= N0:
            # If the number of undecided nodes is less than or equal to N0, set them as coarse nodes
            isC[isU] = True
            isU = []

    return isC, G
