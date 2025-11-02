import pytest
from fealpy.backend import backend_manager as bm
from data_sorting_functions import *

class TestSortingFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("x,axis,descending,stable,exp", argsort_test_data)
    def test_argsort(self, backend, x, axis, descending, stable, exp):
        '''
        Returns the indices that sort an array x along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为后端数组
        x = bm.from_numpy(x)
        exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.argsort(x, axis=axis, descending=descending, stable=stable)
        # 测试
        assert result.shape == exp.shape
        assert bm.all(result == exp)
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("x,axis,descending,stable,exp", sort_test_data)
    def test_sort(self, backend, x, axis, descending, stable, exp):
        '''
        Returns the sorted array x along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为后端数组
        x = bm.from_numpy(x)
        exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.sort(x, axis=axis, descending=descending, stable=stable)
        # 测试
        assert result.shape == exp.shape
        assert bm.all(result == exp)
        

if __name__ == '__main__':
    pytest.main(["test/backend/test_sorting_functions.py", '-qs', '--disable-warnings'])