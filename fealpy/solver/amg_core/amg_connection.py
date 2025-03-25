from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix


def classical_strength_of_connection(Ap, Aj, Ax,n_row, theta=0.25):

    row_inds = bm.repeat(bm.arange(n_row), Ap[1:] - Ap[:-1])
    absAx = bm.abs(Ax)
    
    # 对于非对角元素 (Aj != row_inds) 计算绝对值；对角元素置为0
    is_offdiag = (Aj != row_inds)
    offdiag_abs = absAx.copy()
    offdiag_abs[~is_offdiag] = 0

    # 利用 np.maximum.reduceat 按行计算非对角元素的最大值
    # 注意：每行至少包含一个对角元素，所以该行不会为空
    max_off = bm.maximum.reduceat(offdiag_abs, Ap[:-1])

    # 计算每行的阈值：theta * (最大非对角绝对值)
    threshold = theta * max_off

    keep = (Aj == row_inds) | (absAx >= threshold[row_inds])
    
    # 构造 S 的 CSR 数组（只保留满足条件的元素）
    Sj_new = Aj[keep]
    Sx_new = Ax[keep]
    # 计算每行保留的元素数量（由于 A 本身是按行存储的）
    row_keep = bm.add.reduceat(keep.astype(int), Ap[:-1])
    Sp_new = bm.concatenate(([0], bm.cumsum(row_keep)))


    return Sp_new, Sj_new, Sx_new