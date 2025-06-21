
import ipdb
import numpy as np 
import pytest
from fealpy.backend import backend_manager as bm

# 测试不同的后端
backends = ['numpy', 'pytorch']

class TestBackendInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    def test_unique(self, backend):
        name = ('result', 'indices', 'inverse', 'counts')
        a = bm.array([
            [0, 3], [2, 5], [0, 3], [1, 4], [7, 8], [1, 4]], dtype=bm.int32)

        result = bm.unique(a, return_index=True, 
                           return_inverse=True,
                           return_counts=True, axis=0)

        expected = np.unique(a, return_index=True, 
                             return_inverse=True, 
                             return_counts=True, axis=0)
        print(result)
        print(expected)

        for r, e, s in zip(result, expected, name):
            np.testing.assert_array_equal(bm.to_numpy(r), expected, 
                                          err_msg=f"The {s} of `bm.unique` function is different with numpy result in backend {backend}")

if __name__ == "__main__":
    pytest.main()

