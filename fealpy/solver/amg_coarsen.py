
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

