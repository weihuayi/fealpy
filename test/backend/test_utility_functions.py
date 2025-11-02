import pytest
import numpy as np
from fealpy.backend import backend_manager as bm
from data_utility_functions import *

class TestUtilityFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,exp", all_test_data)
    def test_all(self, backend, input, axis, keepdims, exp):
        '''
        Tests whether all input array elements evaluate to True along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转化为对应的后端数据
        input = bm.from_numpy(input)
        if isinstance(exp, np.ndarray):
            exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.all(input, axis=axis, keepdims=keepdims)
        # 断言测试
        assert result.shape == exp.shape
        assert bm.all(result == exp)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,exp", any_test_data)
    def test_any(self, backend, input, axis, keepdims, exp):
        '''
        Tests whether any input array elements evaluate to True along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转化为对应的后端数据
        input = bm.from_numpy(input)
        if isinstance(exp, np.ndarray):
            exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.any(input, axis=axis, keepdims=keepdims)
        # 断言测试
        assert result.shape == exp.shape
        assert bm.all(result == exp)
    
    #TODO:未实现
    # @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    # @pytest.mark.parametrize("input,axis,n,prepend,append,exp", diff_test_data)
    # def test_diff(self, backend, input, axis, n, prepend, append, exp):
    #     '''
    #     Calculates the n-th discrete forward difference along a specified axis.
    #     '''
    #     # 设置后端
    #     bm.set_backend(backend)
    #     # 转换为对应的后端数据
    #     input = bm.from_numpy(input)
    #     exp = bm.from_numpy(exp)
    #     if isinstance(prepend, np.ndarray):
    #         prepend = bm.from_numpy(prepend)
    #     if isinstance(append, np.ndarray):
    #         append = bm.from_numpy(append)
    #     # 计算结果
    #     result = bm.diff(input, axis=axis, n=n, prepend=prepend, append=append)
    #     # 断言测试
    #     assert result.shape == exp.shape 
    #     assert bm.all(result == exp)
        

if __name__ == '__main__':
    pytest.main(['test/backend/test_utility_functions.py', '-qs', '--disable-warnings'])