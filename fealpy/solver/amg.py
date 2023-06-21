import numpy as np
import scipy.sparse as sp
from timeit import default_timer as timer


class AMGSolver():
    """
    @brief 代数多重网格解法器类。用代数多重网格方法求解
    """
    def __init__(self):
        pass

    def setup(self, A):
        theta = 0.025
        N0 = 50
        NN = np.ceil(np.log2(A.shape[0])/2-4)
        level = max(min( int(NN), 8), 2) # 粗化的层数 
        self.A = [A]
        self.L = [ ] 
        self.U = [ ]
        self.P = [ ]
        self.R = [ ]
        for i in range(level):

            self.L.insert(0, sp.tril(self.A[0])) # 前磨光的光滑子
            self.U.insert(0, sp.triu(self.A[0])) # 后磨光的光滑子

            isC, As = self.coarsen_chen(self.A[0], theta)
            P, R = self.interpolation_standard(As, isC)
            self.A.insert(0, R@self.A[0]@P)
            self.P.insert(0, P)
            self.R.insert(0, R)
            if self.A[0].shape[0] < N0:
                break

        self.R.insert(0, None)



    def solve(self, b):
        pass

    def coarsen_chen(self, A, theta=0.025):
        """
        @brief 

        @param[in] A 对称正定矩阵
        @param[in] theta 粗化阈值
        """

        # 1. 初始化参数
        N = A.shape[0]
        isC = np.zeros(N, dtype=np.bool_)
        N0 = min(int(np.floor(np.sqrt(N))), 25)

        # 2. 生成强连通矩阵 
        # 然后函数计算出归一化的矩阵Am（矩阵A的对角线被归一化），
        # 并找出强连接的节点，也就是那些Am的元素值小于阈值theta的节点。
        # 得到的结果保存在矩阵G中。
        Dinv = sp.diags(1./np.sqrt(A.diagonal()))
        Am = Dinv @ A @ Dinv # 对角线归一化矩阵
        im, jm, sm = sp.find(Am)
        flag = (-sm > theta) 
        # 删除对角、非对角弱联接项，注意对角线元素为1，也会被过滤掉
        G = sp.csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))

        # 3. 计算顶点的度 
        # 函数计算出每个节点的度，也就是与每个节点强连接的节点数量。
        # 如果有太多的节点没有连接，函数会随机选择N0个节点作为粗糙节点并返回。
        deg = np.array(np.sum(sp.csr_matrix(G, dtype=np.bool_), axis=1).flat,
                dtype=np.float64)
        if np.sum(deg > 0) < 0.25*np.sqrt(N):
            isC[np.random.choice(range(N), N0)] = True
            return isC, G

        flag = (deg > 0)
        deg[flag] += 0.1 * np.random.rand(np.sum(flag))

        # 4. 寻找最大独立集 
        # 函数尝试找出一个近似的最大独立集并将其节点添加到粗糙节点集合中。
        # 如果某节点被标记为粗糙节点，则其相邻的节点会被标记为细节点。
        isF = np.zeros(N, dtype=np.bool_)
        isF[deg == 0] = True # 孤立点为细节点 
        isU = np.ones(N, dtype=np.bool_) # 未决定的集合

        while np.sum(isC) < N/2 and np.sum(isU) > N0:
            # 如果粗节点的个数少于总节点个数的一半，并且未决定的点集大于 N0
            isS = np.zeros(N, dtype=np.bool_) # 选择集
            isS[deg>0] = True # 从非孤立点选择
            S = np.nonzero(isS)[0]
            # 非孤立点集的连接关系
            i, j = sp.triu(G[S, :][:, S], 1).nonzero()

            # 第 i 个非孤立点的度大于等于第 j 个非孤立点的度
            flag = deg[S[i]] >= deg[S[j]]
            isS[S[j[flag]] = False # 把度小的节点从选择集移除
            isS[S[i[~flag]]] = False # 把度小的节点从选择集移除
            isC[isS] = True # 剩下的点就是粗点

            # Remove coarse nodes and neighboring nodes from undecided set
            i, _, _ = sp.find(G[:, isC])
            isF[i] = True # 粗点的相邻点是细点
            isU = ~(isF | isC) # 不是细点也不是粗点，就是未决定点
            deg[~isU] = 0 # 粗点或细节的度设置为 0

            if np.sum(isU) <= N0:
                # 如果未决定点的数量小于等于 N0，把未决定点设为粗点
                isC[isU] = True
                isU = []

        return isC, G

    def coarsen_rs(self, A, theta=0.025):
        """
        @brief 
        """
        N = A.shape[0]
        maxaij = A.min(axis=0)
        D = sp.diags(1/np.abs(maxaij).toarray().flatten())
        Am = D @ A

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

            i, j = sp.find(sp.triu(Ass[S][:, S], 1))
            idx = degInAll[S[i]] >= degInAll[S[j]]
            isS[S[j[idx]]] = False
            isS[S[i[~idx]]] = False
            isC[isS] = True

            i, _ = sp.find(Ass[:, isC])
            isF[i] = True
            U = np.where(~(isF | isC))[0]

            degIn[isF | isC] = 0
            degFin = np.zeros(N)
            degFin[U] = np.array(As[isF, U].sum(axis=0)).flatten()

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

        Afc = Am[fineNode, :][:, coarse2fine]
        Dsum = sp.diags(1 / np.array(Afc.sum(axis=1)).flatten())
        ti, tj, tw = sp.find(Dsum @ Afc)
        ip = np.concatenate((ip, fineNode[ti]))
        jp = np.concatenate((jp, tj))
        sp_vals = np.concatenate((sp_vals, tw))
        Pro = sp.coo_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
        Res = Pro.transpose()

        Ac = Res @ A @ Pro
        return Ac, Pro, Res


    def coarsen_a(self, A, theta=0.025):
        """
        @brief 
        """
        N = A.shape[0]
        isC = np.zeros(N, dtype=bool)
        N0 = min(int(np.sqrt(N)), 25)

        # Initialize output
        node2agg = np.zeros(N, dtype=int)
        agg2node = np.zeros(N, dtype=int)

        # Generate strong connectness matrix
        Dinv = sp.diags(1./np.sqrt(A.diagonal()))
        Am = Dinv @ A @ Dinv
        im, jm, sm = sp.find(Am)
        idx = (-sm > theta)
        As = sp.csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
        As += sp.eye(N)
        As1 = sp.csr_matrix(As, dtype=bool)
        As2 = sp.triu(As1 @ As1, 1)

        # Compute degree of vertex
        deg = np.sum(As1, axis=1)
        deg = np.squeeze(np.asarray(deg))
        if np.sum(deg>0) < 0.25*np.sqrt(N):
            isC[np.random.choice(range(N), N0)] = True
            agg2node = np.where(isC)[0]
            node2agg[isC] = np.arange(len(agg2node))
            return node2agg, As

        idx = (deg>0)
        deg[idx] += 0.1 * np.random.rand(np.sum(idx))

        # Find an approximate maximal independent set and put to C set
        isF = np.zeros(N, dtype=bool)
        isU = np.ones(N, dtype=bool)
        isS = np.ones(N, dtype=bool)
        isF[deg == 0] = True
        aggN = 0
        while aggN < N/2 and np.sum(isS) > N0:
            isS = np.zeros(N, dtype=bool)
            isS[deg>0] = True
            S = np.where(isS)[0]
            S_As2 = As2[S,:][:,S]
            i, j = sp.find(S_As2)
            idx = deg[S[i]] >= deg[S[j]]
            isS[S[j[idx]]] = False
            isS[S[i[~idx]]] = False
            isC[isS] = True

            # Add new agg
            newC = np.where(isS)[0]
            newAgg = aggN + np.arange(len(newC))
            aggN += len(newC)
            node2agg[newC] = newAgg
            agg2node[newAgg] = newC

            # Remove coarse nodes and add neighboring nodes to the aggregate
            U = np.where(isU)[0]
            i, j = sp.find(As[isU,:][:,newC])
            isF[U[i]] = True
            isU = ~(isF | isC)
            node2agg[U[i]] = node2agg[newC[j]]
            deg[newC] = 0
            deg[U[i]] = 0
            U = np.where(isU)[0]
            i, _ = sp.find(As[U,:][:,isF])
            deg[U[i]] = 0

        agg2node = agg2node[:max(node2agg)+1]

        # Add left vertices to existing agg
        while any(isU):
            U = np.where(isU)[0]
            i, j = sp.find(As[:, isU])
            neighborAgg = node2agg[i]
            idx = (neighborAgg > 0)
            nAgg, neighborAgg = np.unique(neighborAgg[idx], return_counts=True)
            isbdU = (nAgg > 0)
            bdU = U[isbdU]
            node2agg[bdU] = neighborAgg[isbdU]
            isF[bdU] = True
            isU[bdU] = False

        return node2agg, As


    def interpolation_standard(self, A, isC):
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
        Acf = A[coarseNodeFineIdx, :][:, fineNode] # 把粗-->细块拿出来
        Dsum = np.sum(Acf, axis=0) # 每个细点对应粗点值的和
        flag = (Dsum != 0) # 各非零的
        Nf = np.sum(flag)
        Dsum = sp.diags(1./Dsum[flag], 0)
        ti, tj, tw = sp.find(Acf[:, flag] @ Dsum)
        ip = np.concatenate((coarseNodeFineIdx, fineNode[ti]))
        jp = np.concatenate((coarseNode, tj))
        sp_vals = np.concatenate((np.ones(Nc), tw))
        Pro = sp.csr_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
        Res = Pro.transpose()

        return Pro, Res

    def interpolation_t(self, A, isC):
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

    def interpolation_sa(self, A, node2agg, omega=0.35, smoothingstep=2):
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

    def interpolation_n(self, A, isC):
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




