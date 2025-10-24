import pytest
from fealpy.backend import backend_manager as bm
from data_statistical_functions import *

# max\min\mean\prod\sum\std
class TestStatisticalFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,expected", max_test_data)
    def test_max(self,backend,input,axis,keepdims,expected):
        '''
        Calculates the maximum value of the input array .x
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数据
        input = bm.from_numpy(input)
        if isinstance(expected, np.ndarray):
            expected = bm.from_numpy(expected) 
        # 计算结果
        result = bm.max(input,axis=axis,keepdims=keepdims)
        # 断言结果与预期一致
        if bm.any(bm.isnan(input)):
            assert bm.isnan(result)
        else:
            assert bm.all(result == expected)
    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,expected", min_test_data)
    def test_min(self,backend,input,axis,keepdims,expected):
        '''
        Calculates the minimum value of the input array .x
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数据
        input = bm.from_numpy(input)
        if isinstance(expected, np.ndarray):
            expected = bm.from_numpy(expected) 
        # 计算结果
        result = bm.min(input,axis=axis,keepdims=keepdims)
        # 断言结果与预期一致
        if bm.any(bm.isnan(input)):
            assert bm.isnan(result)
        else:
            assert bm.all(result == expected)
    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,expected", mean_test_data)
    def test_mean(self, backend, input, axis, keepdims, expected):
        
        # 设置后端
        bm.set_backend(backend)
        # 输入转为浮点类型并转换为后端张量
        input = bm.from_numpy(input.astype(np.float64))
        # 将expected转换为后端张量（无论是否为标量）
        if isinstance(expected, np.ndarray):
            # 若预期结果是NumPy数组，转为后端张量
            expected = bm.from_numpy(expected.astype(np.float64))
        else:
            # 若预期结果是Python标量（int/float），先转为NumPy数组再转为后端张量
            expected = bm.from_numpy(np.array(expected, dtype=np.float64))
        
        # 计算结果
        result = bm.mean(input, axis=axis, keepdims=keepdims)
        
        if bm.any(bm.isnan(input)):
            assert bm.isnan(result)
        else:
            assert bm.allclose(result, expected, rtol=1e-6, atol=1e-6)
        
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,expected", prod_test_data)
    def test_prod(self, backend, input, axis, keepdims, expected):
        
        # 设置后端
        bm.set_backend(backend)
        # 处理 axis == None 的情况
        if axis == None:
            axis = -1
        # 输入转为浮点类型并转换为后端张量
        input = bm.from_numpy(input.astype(np.float64))
        # 将expected转换为后端张量（无论是否为标量）
        if isinstance(expected, np.ndarray):
            # 若预期结果是NumPy数组，转为后端张量
            expected = bm.from_numpy(expected.astype(np.float64))
        else:
            # 若预期结果是Python标量（int/float），先转为NumPy数组再转为后端张量
            expected = bm.from_numpy(np.array(expected, dtype=np.float64))
        
        # 计算结果
        result = bm.prod(input, axis=axis, keepdims=keepdims)
        
        if bm.any(bm.isnan(input)):
            assert bm.isnan(result)
        else:
            assert bm.allclose(result, expected, rtol=1e-6, atol=1e-6)

    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,keepdims,expected", sum_test_data)
    def test_sum(self, backend, input, axis, keepdims, expected):
        
        # 设置后端
        bm.set_backend(backend)
        # 处理 axis == None 的情况
        if axis == None:
            axis = -1
        # 输入转为浮点类型并转换为后端张量
        input = bm.from_numpy(input.astype(np.float64))
        # 将expected转换为后端张量（无论是否为标量）
        if isinstance(expected, np.ndarray):
            # 若预期结果是NumPy数组，转为后端张量
            expected = bm.from_numpy(expected.astype(np.float64))
        else:
            # 若预期结果是Python标量（int/float），先转为NumPy数组再转为后端张量
            expected = bm.from_numpy(np.array(expected, dtype=np.float64))
        
        # 计算结果
        result = bm.sum(input, axis=axis, keepdims=keepdims)
        
        if bm.any(bm.isnan(input)):
            assert bm.isnan(result)
        else:
            assert bm.allclose(result, expected, rtol=1e-10, atol=1e-6)
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("input,axis,correction,keepdims,expected", std_test_data)
    def test_std(self, backend, input, axis, correction, keepdims, expected):   
        
        # 设置后端
        bm.set_backend(backend)
        correction = correction
        # 输入转为浮点类型并转换为后端张量
        input = bm.from_numpy(input.astype(np.float64))
        # 将expected转换为后端张量（无论是否为标量）
        if isinstance(expected, np.ndarray):
            # 若预期结果是NumPy数组，转为后端张量
            expected = bm.from_numpy(expected.astype(np.float64))
        else:
            # 若预期结果是Python标量（int/float），先转为NumPy数组再转为后端张量
            expected = bm.from_numpy(np.array(expected, dtype=np.float64))
        
        # 计算结果
        result = bm.std(input, axis=axis, correction=correction, keepdims=keepdims)
    
        # 用bm.any()将布尔张量转为单个布尔值
        if bm.any(bm.isnan(expected)):
            assert bm.any(bm.isnan(result))  # 确保结果中至少有一个 nan
        # 判断输入是否含 nan（输入含 nan 时结果必为 nan）
        elif bm.any(bm.isnan(input)):
            assert bm.any(bm.isnan(result))
        # 3. 正常数值对比
        else:
            assert bm.allclose(result, expected, rtol=1e-6, atol=1e-6)
            

if __name__ == '__main__':
    pytest.main(['test/backend/test_statistical_functions.py', '-qs', '--disable-warnings'])
