import pytest
from fealpy.backend import backend_manager as bm
from data_searching_functions import *


class TestSearchingFunctionsInterfaces:

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("condition,x1,x2,exp", where_test_data)
    def test_where(self, backend, condition, x1, x2, exp):
        """
        Returns elements chosen from x1 or x2 depending on condition.
        """
        # 设置后端
        try:
            bm.set_backend(backend)
        except Exception as e:
            pytest.skip(f"backend {backend} not available: {e}")
        # 转换为对应的后端数组
        condition = bm.from_numpy(condition)
        exp = bm.from_numpy(exp)  # 计算结果

        if type(x1) == np.ndarray:
            x1 = bm.from_numpy(x1)
        if type(x2) == np.ndarray:
            x2 = bm.from_numpy(x2)

        result = bm.where(condition, x1, x2)
        # 测试
        assert bm.all(result == exp)

    @pytest.mark.parametrize("backend", ["numpy", "pytorch"])
    @pytest.mark.parametrize("x, exp", nonzero_test_data)
    def test_nonzero(self, backend, x, exp):
        """
        Returns the indices of the array elements which are non-zero.
        """
        # 设置后端
        try:
            bm.set_backend(backend)
        except Exception as e:
            pytest.skip(f"backend {backend} not available: {e}")
        # 转化为对应的后端数组
        x = bm.from_numpy(x)
        exp = tuple(bm.from_numpy(i) for i in exp)
        # 计算结果
        result = bm.nonzero(x)
        # 测试
        for i, j in zip(result, exp):
            assert bm.all(i == j)


if __name__ == "__main__":
    pytest.main(
        ["test/backend/test_searching_functions.py", "-qs", "--disable-warnings"]
    )
