from typing import Callable
from scipy.sparse import csr_matrix
from torch import Tensor
from numpy.typing import NDArray
import numpy as np


def enable_csr(fn: Callable[..., Tensor]):
    def wrapped(self, return_sparse: bool=False):
        ret = fn()
        if return_sparse:
            data: NDArray = ret.numpy()
            if data.ndim != 2:
                raise ValueError("Can only tackle tensors with 2 dimension.")
            nr, nc = data.shape
            sp = csr_matrix(
                (
                    np.ones(nr*nc, dtype=np.bool_),
                    (
                        np.repeat(range(nr), nc),
                        data.flat
                    )
                ),
                shape=(nr, nc)
            )
            return sp

        else:
            return ret

    return wrapped
