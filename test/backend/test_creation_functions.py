import pytest
from fealpy.backend import backend_manager as bm
from data_creation_functions import *

class TestCreationFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("start,stop,step,expected", arange_test_data)
    def test_arange(self, backend, start, stop, step, expected):
        '''
        Returns evenly spaced values within the half-open interval [start, stop) as a one-dimensional array.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 解决浮点数精度问题
        is_float = expected.dtype.kind == 'f'
        # pytorch 不支持 None 类型，使用默认值代替
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1
        # 转换为后端数组
        expected = bm.from_numpy(expected)
        # 测试
        if is_float:
            assert bm.allclose(bm.arange(start, stop, step, dtype=expected.dtype), expected)
        else:
            assert bm.all(bm.arange(start, stop, step) == expected)
            
            
if __name__ == '__main__':
    pytest.main(['test/backend/test_creation_functions.py', '-qs', '--disable-warnings'])