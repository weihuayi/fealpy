
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags

ALL_BACKENDS = ['numpy', 'pytorch']


class test_spdiags():
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_scalar_diags(self, backend):
        bm.set_backend(backend)
        diags = -1
        data = bm.array([1,2,3,4],dtype=bm.float64)

        csr_diags = spdiags(data, diags, 4, 4, format='csr')
        coo_diags = spdiags(data, diags, 4, 4, format='coo')

        expected_diag = bm.array([[0, 0, 0, 0], 
                                  [1, 0, 0, 0], 
                                  [0, 2, 0, 0], 
                                  [0, 0, 3, 0]], dtype=data.dtype)

        assert bm.allclose(csr_diags.toarray(), expected_diag)
        assert bm.allclose(coo_diags.toarray(), expected_diag)

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_tensor_spdiags(self, backend):
        bm.set_backend(backend)
        diags = bm.array([0, 1, -2], dtype=bm.int64)
        data = bm.array([[1, 2, 3, 4], [1, 2, 3, -4.0], [1, 2, 3, 4]], dtype=bm.float64)

        csr_diags = spdiags(data, diags, 4, 4, format='csr')
        coo_diags = spdiags(data, diags, 4, 4, format='coo')

        expected_diag = bm.array([[1, 2, 0, 0], 
                                  [0, 2, 3, 0], 
                                  [1, 0, 3, -4], 
                                  [0, 2, 0, 4]], dtype=data.dtype)

        assert bm.allclose(csr_diags.toarray(), expected_diag)
        assert bm.allclose(coo_diags.toarray(), expected_diag)

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_spdiags_shape(self, backend):
        bm.set_backend(backend)
        diags = bm.array([0, 1], dtype=bm.int64)
        data = bm.array([[1, 2, 3], [1, 2, 3]], dtype=bm.float64)

        csr_diags = spdiags(data, diags, 4, 5, format='csr')
        coo_diags = spdiags(data, diags, 4, 5, format='coo')

        expected_diag = bm.array([[1, 2, 0, 0, 0], 
                                  [0, 2, 3, 0, 0], 
                                  [0, 0, 3, 0, 0], 
                                  [0, 0, 0, 0, 0]], dtype=data.dtype)

        assert bm.allclose(csr_diags.toarray(), expected_diag)
        assert bm.allclose(coo_diags.toarray(), expected_diag)
