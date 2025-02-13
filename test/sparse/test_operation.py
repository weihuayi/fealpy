
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.sparse.operation import spdiags 
ALL_BACKENDS = ['numpy', 'pytorch']

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_spdiags(backend):
    bm.set_backend(backend)

    M = 4
    N = 4
    diags = bm.array([0, 1, -2])
    data = bm.array([[1, 2, 3, 4], [1, 2, 3, -4.0], [1, 2, 3, 4]])

    csr_diags = spdiags(data, diags, M, N, format='csr')
    coo_diags = spdiags(data, diags, M, N, format='coo')

    expected_diag = bm.array([[1, 2, 0, 0],
                              [0, 2, 3, 0],
                              [1, 0, 3, -4],
                              [0, 2, 0, 4]], dtype=data.dtype)

    assert bm.allclose(csr_diags.toarray(), expected_diag)
    assert bm.allclose(coo_diags.toarray(), expected_diag)
