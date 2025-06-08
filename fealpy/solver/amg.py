from .amg_core import classical_strength_of_connection, rs_cf_splitting,rs_direct_interpolation
from ..sparse import csr_matrix
from ..backend import backend_manager as bm

def ruge_stuben_amg(A,theta = 0.25):
    
    n_row = A.shape[0]
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Sp, Sj, Sx = classical_strength_of_connection(Ap, Aj, Ax,n_row,theta)
    S = csr_matrix((Sx, Sj, Sp), shape=(n_row, n_row))
    T = S.T
    Tp, Tj = T.indptr, T.indices
    splitting = rs_cf_splitting(Sp, Sj, Tp, Tj, n_row)
    p,r = rs_direct_interpolation(Ap, Aj, Ax,Sp, Sj, Sx, splitting)
    return p,r

# def ruge_stuben_coarse(A,theta = 0.025):
#     isC,Am = ruge_stuben_chen_coarsen(A,theta)
#     p,r = two_points_interpolation(A,isC)
#     return p,r
