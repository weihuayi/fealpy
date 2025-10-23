import pytest
import numpy as np
from fealpy.backend import backend_manager as bm
from data_set_functions import unique_all_test_data

class TestSetFunctionsInterfaces:

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("x, exp", unique_all_test_data)
    def test_unique_all(self, backend, x, exp):
        '''
        Returns the unique elements of an input array x, 
        the first occurring indices for each unique element in x, 
        the indices from the set of unique elements that reconstruct x, 
        and the corresponding counts for each unique element in x.
        '''
        # PyTorch 不支持复数 unique，跳过
        if backend == "pytorch" and np.iscomplexobj(x):
            pytest.skip("PyTorch backend does not support unique for complex dtypes.")

        try:
            bm.set_backend(backend)
        except Exception as e:
            pytest.skip(f"backend {backend} not available: {e}")

        x = bm.from_numpy(x)
        exp = tuple(bm.from_numpy(i) for i in exp)
        # 计算结果
        result = bm.unique_all(x)
        # 测试
        for i, j in zip(result, exp):
            assert bm.allclose(i, j, equal_nan=True)

if __name__ == "__main__":
    pytest.main(["test/backend/test_set_functions.py", "-qs", "--disable-warnings"])