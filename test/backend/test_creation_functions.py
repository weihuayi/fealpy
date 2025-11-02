import pytest
from fealpy.backend import backend_manager as bm
from data_creation_functions import *

class TestCreationFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("start,stop,step,exp", arange_test_data)
    def test_arange(self, backend, start, stop, step, exp):
        '''
        Returns evenly spaced values within the half-open interval [start, stop) as a one-dimensional array.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 解决浮点数精度问题
        is_float = exp.dtype.kind == 'f'
        # pytorch 不支持 None 类型，使用默认值代替
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1
        # 转换为对应的后端数组
        exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.arange(start, stop, step, dtype=exp.dtype)
        # 测试
        assert result.shape == exp.shape
        if is_float:
            assert bm.allclose(result, exp)
        else:
            assert bm.all(result == exp)
            
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("x,dtype,exp", asarray_test_data)
    def test_asarray(self, backend, dtype_map, x, dtype, exp):
        '''
        Converts the input to an array.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数组
        if isinstance(x, np.ndarray):
            x = bm.from_numpy(x)
        exp = bm.from_numpy(exp)
        if dtype is not None:
            dtype = dtype_map[backend][dtype]
        # 计算结果
        result = bm.asarray(x, dtype=dtype)
        # 测试
        assert result.shape == exp.shape
        assert bm.allclose(result, exp)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_empty(self, backend, backend_map):
        '''
        Returns an uninitialized array having a specified shape.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.empty` 指向正确的底层实现函数。
        module_backend = backend_map[backend]
        assert bm.empty is module_backend.empty
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_empty_like(self, backend, backend_map):
        '''
        Returns an uninitialized array with the same shape as an input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.empty` 指向正确的底层实现函数。
        module_backend = backend_map[backend]
        assert bm.empty_like is module_backend.empty_like
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("n_rows,n_cols,k,exp", eye_test_data)
    def test_eye(self, backend, n_rows, n_cols, k, exp):
        '''
        Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转化为对应的后端数组
        exp = bm.from_numpy(exp)
        # 计算结果
        result = bm.eye(n_rows, n_cols, k)
        # 测试
        assert result.shape == exp.shape
        assert bm.all(result == exp)
            
if __name__ == '__main__':
    pytest.main(['test/backend/test_creation_functions.py', '-qs', '--disable-warnings'])