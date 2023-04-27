from typing import Callable
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import numpy as np


def enable_csr(fn: Callable[..., NDArray]):
    """
    @brief Make a function generating neighborhood infomation matrix\
           supporting csr matrix output.
    """
    def wrapped(self, *args, return_sparse: bool=False, **kwargs):
        ret = fn(self, *args, **kwargs)
        if return_sparse:
            if ret.ndim != 2:
                raise ValueError("Can only tackle tensors with 2 dimension.")
            nr, nc = ret.shape
            sp = csr_matrix(
                (
                    np.ones(nr*nc, dtype=np.bool_),
                    (
                        np.repeat(range(nr), nc),
                        ret.flat
                    )
                ),
                shape=(nr, nc)
            )
            return sp

        else:
            return ret

    return wrapped
