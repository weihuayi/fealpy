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


def symmetric_strength_of_connection( Ap, Aj, Ax,n_row, theta = 0.25):
    """
    计算对称强连接矩阵 S，基于经典的对称 SA 强度度量，
    使用向量化操作避免 for 循环。
    
    对于 CSR 格式矩阵 A，其非零条目 A[i,j] 被认为是强连接当且仅当：
    
        |A[i,j]| >= theta * sqrt(|A[i,i]| * |A[j,j]|)
        
    等价于：
    
        |A[i,j]|^2 >= theta^2 * |A[i,i]| * |A[j,j]|
        
    对角条目总保留。
    
    参数：
      n_row : int
          矩阵 A 的行数。
      theta : float
          阈值参数（正数）。
      Ap : array_like
          CSR 格式的行指针（长度 n_row+1）。
      Aj : array_like
          CSR 格式的列索引。
      Ax : array_like
          CSR 格式的非零值数组。
    
    返回：
      S : csr_matrix
          强连接矩阵（CSR 格式），形状 (n_row, n_row)。
    """
    
    # 1. 构造每个非零条目所属行号：row_inds，长度等于 len(Ax)
    row_inds = bm.repeat(bm.arange(n_row), Ap[1:] - Ap[:-1])
    
    # 2. 计算对角元素值：对于每个 i，取 A[i,i]（可能有重复，则累加）
    diag_mask = (Aj == row_inds)
    # 使用 np.bincount 累加每行的对角值，注意 minlength 确保长度正确
    diags = bm.bincount(row_inds[diag_mask], weights=Ax[diag_mask], minlength=n_row)
    diags = bm.abs(diags)  # 取绝对值作为尺度
    
    # 3. 对于所有条目，计算条件
    # 条件：对于非对角条目: |A[i,j]|^2 >= theta^2 * diags[i] * diags[j]
    # 对角条目总保留
    absAx = bm.abs(Ax)
    # 这里计算平方条件
    cond_non_diag = (absAx**2) >= (theta**2) * diags[row_inds] * diags[Aj]
    # 构造保留 mask：对角条目或满足条件的非对角条目
    keep_mask = (row_inds == Aj) | cond_non_diag
    
    # 4. 筛选保留的条目
    Sj_new = Aj[keep_mask]
    Sx_new = Ax[keep_mask]
    
    # 5. 构造新的 CSR 行指针：统计每一行保留的个数
    row_counts = bm.bincount(row_inds[keep_mask], minlength=n_row)
    Sp_new = bm.concatenate(([0], bm.cumsum(row_counts)))
    
    
    return Sp_new, Sj_new, Sx_new