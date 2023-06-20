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
        """
        N = A.shape[0]
        isC = np.zeros(N, dtype=bool)
        N0 = min(int(np.floor(np.sqrt(N))), 25)

        # Generate strong connectness matrix
        Dinv = sp.diags(1./np.sqrt(A.diagonal()))
        Am = Dinv @ A @ Dinv
        im, jm, sm = sp.find(Am)
        idx = (-sm > theta)
        G = sp.csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))

        # Compute degree of vertex
        deg = np.sum(sp.csr_matrix(G, dtype=np.bool_), axis=1)
        deg = np.squeeze(np.asarray(deg))
        if np.sum(deg>0) < 0.25*np.sqrt(N):
            isC[np.random.choice(range(N), N0)] = True
            return isC, G

        idx = (deg>0)
        deg[idx] += 0.1 * np.random.rand(np.sum(idx))

        # Find an approximated maximal independent set and put to C set
        isF = np.zeros(N, dtype=bool)
        isF[deg == 0] = True
        isU = np.ones(N, dtype=bool)

        while np.sum(isC) < N/2 and np.sum(isU) > N0:
            isS = np.zeros(N, dtype=bool)
            isS[deg>0] = True
            S = np.where(isS)[0]
            S_G = G[S,:][:,S]
            i,j = sp.triu(S_G,1).nonzero()
            idx = deg[S[i]] >= deg[S[j]]
            isS[S[j[idx]]] = False
            isS[S[i[~idx]]] = False
            isC[isS] = True

            # Remove coarse nodes and neighboring nodes from undecided set
            i, _, _ = sp.find(G[:,isC])
            isF[i] = True
            isU = ~(isF | isC)
            deg[~isU] = 0

            if np.sum(isU) <= N0:
                isC[isU] = True
                isU = []

        # print('Number of Coarse Nodes:', np.sum(isC))
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
        @brief 
        """
        N = A.shape[0]

        # Index map between coarse grid and fine grid
        allNode = np.arange(N)
        fineNode = allNode[~isC]
        Nc = N - len(fineNode)
        coarseNode = np.arange(Nc)
        coarseNodeFineIdx = np.where(isC)[0]

        # Construct prolongation and restriction operator
        Acf = A[coarseNodeFineIdx,:][:,fineNode]
        Dsum = np.sum(Acf, axis=0)
        idx = (Dsum != 0)
        Nf = np.sum(idx)
        Dsum = sp.diags(1./Dsum[idx], 0)
        ti, tj, tw = sp.find(Acf[:,idx] @ Dsum)
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




