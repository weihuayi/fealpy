#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ****************************************************
# @FileName      :   test_taichi_backend.py
# @Time          :   2025/06/21 13:22:04
# @Author        :   XuMing
# @Version       :   1.0
# @Email         :   920972751@qq.com
# @Description   :   None
# @Copyright     :   XuMing. All Rights Reserved.
# ****************************************************


import taichi as ti
import pytest

from fealpy.backend import backend_manager as bm
import numpy as np
bm.set_backend('taichi')



# 测试 set_default_device 方法
def test_set_default_device():
    # 测试设置为 'cpu'
    bm.set_default_device('cpu')
    assert bm.get_current_backend()._device == ti.cpu

    # # 测试设置为 'cuda'
    # bm.set_default_device('cuda')
    # assert bm.get_current_backend()._device == ti.cuda

    # 测试设置为 ti.cpu
    bm.set_default_device(ti.cpu)
    assert bm.get_current_backend()._device == ti.cpu

    # # 测试设置为 ti.cuda
    # bm.set_default_device(ti.cuda)
    # assert bm.get_current_backend()._device == ti.cuda

# 测试 context 方法
def test_context():
    x = ti.field(dtype=ti.f32, shape=(2, 3))
    # 填充数据
    @ti.kernel
    def fill():
        for i, j in x:
            x[i, j] = i * 1.0 + j * 0.1
    fill()

    ctx = bm.context(x)
    # dtype 应为 Taichi DataType
    assert ctx['dtype'] == x.dtype
   
    # device 应为 'cpu'
    assert ctx['device'] == 'cpu'
    
    print(ctx)
    
# 测试 to_numpy 方法
def test_to_numpy():
    x = ti.field(dtype=ti.f32, shape=(2, 3))
    # 填充数据
    @ti.kernel
    def fill():
        for i, j in x:
            x[i, j] = i * 1.0 + j * 0.1
    fill()

    print("taichi data type: ", x.dtype)  # 输出 Taichi 的 dtype
    np_x = bm.to_numpy(x)
    print("taichi data to numpy type: ", np_x.dtype)
    assert isinstance(np_x, np.ndarray)
    assert np.allclose(np_x, np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]))
    
# 测试 from_numpy 方法
class TestFromNumpy:
    def test_from_numpy_float32(self):
        # 初始化Taichi（如果尚未初始化）
        ti.init(arch=ti.cpu)  # 使用CPU后端加速测试
        
        np_array = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        ti_field = bm.from_numpy(np_array)
        
        # 检查数据类型
        assert ti_field.dtype == ti.f32
        
        # 检查形状
        assert ti_field.shape == (3,)
        
        # 检查数据内容（允许浮点数微小误差）
        assert np.allclose(ti_field.to_numpy(), np_array)
        
    def test_from_numpy_int32(self):
        # 初始化Taichi
        ti.init(arch=ti.cpu)  # 使用CPU后端加速测试
        
        np_array = np.array([1, 2, 3], dtype=np.int32)
        ti_field = bm.from_numpy(np_array)
        
        # 检查数据类型
        assert ti_field.dtype == ti.i32
        
        # 检查形状
        assert ti_field.shape == (3,)
        
        # 检查数据内容
        assert np.allclose(ti_field.to_numpy(), np_array)



if __name__ == '__main__':
    pytest.main(['-q', '-s'])

