"""Cellwise operations on matrices sharing one CSR sparsity pattern."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor


def spmv(A: CSRTensor, x):
    """Compute ``y[K] = A[K] @ x[K]`` for every cell ``K``."""
    nrow = A.sparse_shape[0]
    y = bm.zeros((x.shape[0], nrow), dtype=x.dtype, device=bm.get_device(x))
    for i in range(nrow):
        start, end = int(A.crow[i]), int(A.crow[i+1])
        col = A.col[start:end]
        value = bm.sum(A.values[:, start:end] * x[:, col], axis=1)
        y = bm.set_at(y, (slice(None), i), value)
    return y


def spsolve_triangular(A: CSRTensor, b, lower=True, unit_diagonal=False):
    """Solve ``A[K] x[K] = b[K]`` cell by cell.

    ``A`` is nonsingular and triangular. With ``unit_diagonal=True``, its
    diagonal is treated as one. ``b`` may have shape ``(NC, n)`` or
    ``(NC, n, nrhs)`` and is not modified.
    """
    vector_rhs = b.ndim == 2
    if vector_rhs:
        b = b[..., None]

    n = A.sparse_shape[0]
    x = bm.zeros_like(b)
    rows = range(n) if lower else range(n-1, -1, -1)
    for i in rows:
        start, end = int(A.crow[i]), int(A.crow[i+1])
        col = A.col[start:end]
        values = A.values[:, start:end]
        off_diagonal = col < i if lower else col > i
        value = b[:, i, :] - bm.sum(
            values[:, off_diagonal, None] * x[:, col[off_diagonal], :], axis=1
        )
        if not unit_diagonal:
            diagonal = bm.sum(values[:, col == i], axis=1)
            value = value / diagonal[:, None]
        x = bm.set_at(x, (slice(None), i, slice(None)), value)
    return x[..., 0] if vector_rhs else x


def transpose(A: CSRTensor):
    """Transpose a shared CSR pattern and permute its cellwise values."""
    nrow, ncol = A.sparse_shape
    entries = []
    for row in range(nrow):
        start, end = int(A.crow[row]), int(A.crow[row+1])
        entries.extend((int(A.col[k]), row, k) for k in range(start, end))
    entries.sort()
    counts = [0]*ncol
    for row, _, _ in entries:
        counts[row] += 1
    crow = [0]
    for count in counts:
        crow.append(crow[-1]+count)
    col = bm.array([col for _, col, _ in entries], dtype=bm.int32)
    position = bm.array([k for _, _, k in entries], dtype=bm.int32)
    values = A.values[..., position]
    return CSRTensor(bm.array(crow, dtype=bm.int32), col, values, (ncol, nrow))


def sparse_congruence(A: CSRTensor, M):
    """Compute cellwise ``A.T @ M @ A`` without densifying ``A``."""
    nc, n, _ = M.shape
    values = A.values
    if values.ndim == 1:
        values = values[None, :]
    right = bm.zeros_like(M)
    for row in range(n):
        start, end = int(A.crow[row]), int(A.crow[row+1])
        col = A.col[start:end]
        update = M[:, :, row, None]*values[:, None, start:end]
        right = bm.set_at(
            right, (slice(None), slice(None), col),
            right[:, :, col]+update
        )

    result = bm.zeros_like(M)
    for row in range(n):
        start, end = int(A.crow[row]), int(A.crow[row+1])
        col = A.col[start:end]
        update = values[:, start:end, None]*right[:, row, None, :]
        result = bm.set_at(
            result, (slice(None), col, slice(None)),
            result[:, col, :]+update
        )
    return result
