
from ...backend import bm, Tensor, dtype


def argpermute(src: Tensor, tgt: Tensor, /, *, dtype: dtype | None = None) -> Tensor:
    """Return the indices that permute the source array to the target array.

    The source and target arrays must have the same shape and contain the same
    elements, but they may be in different orders.

    For multiple dimensions, the permutation is applied to the last dimensions
    of the source array."""
    # Sort both arrays on the last axis. Then map each target position to the
    # source index at the same rank in sorted order.
    src_sorted_to_src = bm.argsort(src, axis=-1, stable=True)

    if dtype is not None:
        src_sorted_to_src = bm.asarray(src_sorted_to_src, dtype=dtype)

    tgt_sorted_to_tgt = bm.argsort(tgt, axis=-1, stable=True)
    tgt_to_tgt_sorted = bm.argsort(tgt_sorted_to_tgt, axis=-1, stable=True)

    del tgt_sorted_to_tgt

    if dtype is not None:
        tgt_to_tgt_sorted = bm.asarray(tgt_to_tgt_sorted, dtype=dtype)

    return bm.take_along_axis(src_sorted_to_src, tgt_to_tgt_sorted, axis=-1)
