from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags
from fealpy.sparse import csr_matrix

def ruge_stuben_interpolation(isC,Am):
    N = Am.shape[0]
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

def standard_interpolation(A, isC):
    """
    @brief Generate prolongation and restriction matrices

    @param[in] A A symmetric positive definite matrix
    @param[in] isC A boolean array marking the coarse points

    @note The Prolongation matrix interpolates the solution from the coarse grid to the fine grid;
          The Restriction matrix restricts the residual from the fine grid to the coarse grid.
    """

    N = A.shape[0]

    # 1. Index mapping: The function first creates an index mapping from the coarse grid to the fine grid.
    #    It identifies all coarse and fine nodes and stores their indices.
    allNode = bm.arange(N)
    fineNode = allNode[~isC]
    NC = N - len(fineNode)
    coarseNode = bm.arange(NC)
    coarseNodeFineIdx = bm.nonzero(isC)[0]

    # 2. Construct prolongation and restriction operators
    Acf = A[coarseNodeFineIdx, fineNode]  # Extract the coarse-to-fine matrix block
    Dsum = bm.asarray(Acf.sum(axis=0)).reshape(-1)  # Sum of values corresponding to each fine node
    flag = (Dsum != 0)  # Boolean array marking fine nodes with nonzero sums
    NF = bm.sum(flag)  # Number of fine nodes
    Dsum = spdiags(1./Dsum[flag], diags=0, M=NF, N=NF)  # Form a sparse diagonal matrix
    flag = bm.nonzero(flag)[0]
    i, j, w = (Acf[:, flag] @ Dsum).find()  # Normalize each column by its sum
    # Note: 'j' represents fine node indices, 'i' represents coarse node indices
    # The prolongation matrix transfers information from the coarse grid to the fine grid
    I = bm.concatenate((coarseNodeFineIdx, fineNode[j]))
    J = bm.concatenate((coarseNode, i))
    val = bm.concatenate((bm.ones(NC), w))
    P = csr_matrix((val, (I, J)), shape=(N, NC))
    R = P.T
    return P, R
