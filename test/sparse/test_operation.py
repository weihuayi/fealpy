
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.sparse.ops import spdiags, speye

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

class test_speye():
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_scalar_diags(self, backend):
        bm.set_backend(backend)

        dtype = bm.float64

        csr_diags1 = speye(4, format='csr', dtype=dtype)
        coo_diags1 = speye(4, format='coo', dtype=dtype)

        diags = -1
        csr_diags2 = speye(4, diags=diags, format='csr', dtype=dtype)
        coo_diags2 = speye(4, diags=diags, format='coo', dtype=dtype)

        diags = 1
        csr_diags3 = speye(4, 5, diags=diags, format='csr', dtype=dtype)
        coo_diags3 = speye(4, 5, diags=diags, format='coo', dtype=dtype)

        expected_diag1 = bm.array([[1, 0, 0, 0], 
                                  [0, 1, 0, 0], 
                                  [0, 0, 1, 0], 
                                  [0, 0, 0, 1]], dtype=dtype)

        expected_diag2 = bm.array([[0, 0, 0, 0], 
                                  [1, 0, 0, 0], 
                                  [0, 1, 0, 0], 
                                  [0, 0, 1, 0]], dtype=dtype)

        expected_diag3 = bm.array([[0, 1, 0, 0, 0], 
                                  [0, 0, 1, 0, 0], 
                                  [0, 0, 0, 1, 0], 
                                  [0, 0, 0, 0, 0]], dtype=dtype)

        assert bm.allclose(csr_diags1.toarray(), expected_diag1)
        assert bm.allclose(coo_diags1.toarray(), expected_diag1)
        assert bm.allclose(csr_diags2.toarray(), expected_diag2)
        assert bm.allclose(coo_diags2.toarray(), expected_diag2)
        assert bm.allclose(csr_diags3.toarray(), expected_diag3)
        assert bm.allclose(coo_diags3.toarray(), expected_diag3)

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_tensor_speye(self, backend):
        bm.set_backend(backend)
        dtype = bm.float64
        diags = bm.array([0, 1, -2])

        csr_diags = speye(4, 4, diags, format='csr', dtype=dtype)
        coo_diags = speye(4, 4, diags, format='coo', dtype=dtype)

        expected_diag = bm.array([[1, 1, 0, 0], 
                                  [0, 1, 1, 0], 
                                  [1, 0, 1, 1], 
                                  [0, 1, 0, 1]], dtype=dtype)

        assert bm.allclose(csr_diags.toarray(), expected_diag)
        assert bm.allclose(coo_diags.toarray(), expected_diag)