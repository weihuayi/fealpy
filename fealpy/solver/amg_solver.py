from .amg_core.amg_connection import classical_strength_of_connection
from .amg_core.amg_splitting import rs_cf_splitting
from .amg_core.amg_interpolation import rs_direct_interpolation
from fealpy.sparse import csr_matrix
from ..backend import backend_manager as bm

def csr_transpose(indptr, indices,data, shape):
    m, n = shape
    # 生成原矩阵的每个非零元素对应的行索引（向量化）
    row_indices = bm.repeat(bm.arange(m), indptr[1:] - indptr[:-1])
    
    # 按列索引排序（稳定排序保证相同列内保持原顺序）
    order = bm.argsort(indices)
    
    new_indices = row_indices[order]
    new_data = data[order]
    # 统计原矩阵每个列的非零个数，并构造新的 indptr 数组
    col_counts = bm.bincount(indices, minlength=n)
    new_indptr = bm.empty(n + 1, dtype=int)
    new_indptr[0] = 0
    new_indptr[1:] = bm.cumsum(col_counts)
    
    return new_indptr, new_indices, new_data


def Ruge_Stuben_AMG(A,theta = 0.25):
    
    n_row = A.shape[0]
    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Sp, Sj, Sx = classical_strength_of_connection(Ap, Aj, Ax,n_row,theta)
    Tp, Tj, _ = csr_transpose(Sp, Sj, Sx, (n_row, n_row))
    splitting = rs_cf_splitting(Sp, Sj, Tp, Tj, n_row)
    p,r = rs_direct_interpolation(Ap, Aj, Ax,Sp, Sj, Sx, splitting)
    # Px, Pj, Pp, n_coarse = rs_direct_interpolation(Ap, Aj, Ax,Sp, Sj, Sx, splitting)
    # rp,rj,rx = csr_transpose(Pp, Pj, Px, (n_row, n_coarse))
    
    
    return p,r