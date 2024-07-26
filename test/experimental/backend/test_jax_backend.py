
import pytest

import jax
import jax.numpy as jnp

from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('jax')

def test_multi_index_matrix():
    m = bm.multi_index_matrix(2, 2)
    print(m)


if __name__ == "__main__":
    test_multi_index_matrix()

