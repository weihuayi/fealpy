import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace.functional import *

from functional_data import *


class TestFunctional:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("symmetry_index_data", symmetry_index_data)
    def test_symmetry_index(self, symmetry_index_data, backend):
        bm.set_backend(backend)
        d = symmetry_index_data['d']
        r = symmetry_index_data['r']
        symidx_true = symmetry_index_data['symidx']
        num_true    = symmetry_index_data['num']

        symidx, num = symmetry_index(d, r)

        np.testing.assert_array_equal(bm.to_numpy(symidx), symidx_true)
        np.testing.assert_array_equal(bm.to_numpy(num), num_true)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("symmetry_span_array_data", symmetry_span_array_data)
    def test_symmetry_span_array(self, symmetry_span_array_data, backend):
        bm.set_backend(backend)

        t = bm.tensor(symmetry_span_array_data['t'])
        alpha = bm.tensor(symmetry_span_array_data['alpha'])
        symt_true  = symmetry_span_array_data['symt']

        symt = symmetry_span_array(t, alpha)

        np.testing.assert_allclose(bm.to_numpy(symt), symt_true, atol=1e-14)


if __name__ == '__main__':
    pytest.main()


