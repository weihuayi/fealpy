import pytest
from fealpy.backend import backend_manager as bm
from data_sorting_functions import *

class TestSortingFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("x,axis,descending,stable,expected", argsort_test_data)
    def test_argsort(self, backend, x, axis, descending, stable, expected):
        '''
        Returns the indices that sort an array x along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为后端数组
        x = bm.from_numpy(x)
        expected = bm.from_numpy(expected)
        # 测试
        assert bm.all(bm.argsort(x, axis=axis, descending=descending, stable=stable) == expected)
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("x,axis,descending,stable,expected", sort_test_data)
    def test_sort(self, backend, x, axis, descending, stable, expected):
        '''
        Returns the sorted array x along a specified axis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为后端数组
        x = bm.from_numpy(x)
        expected = bm.from_numpy(expected)
        # 测试
        assert bm.all(bm.sort(x, axis=axis, descending=descending, stable=stable) == expected)
        

if __name__ == '__main__':
    pytest.main(["test/backend/test_sorting_functions.py", '-qs', '--disable-warnings'])