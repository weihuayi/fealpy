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

#测试 device_type 方法
def test_device_type():
    x = ti.field(dtype=ti.f32, shape=(2, 3))

    # 填充数据
    @ti.kernel
    def fill():
        for i, j in x:
            x[i, j] = i * 1.0 + j * 0.1

    fill()

    dt = bm.device_type(x)

    # device 应为 'cpu'
    assert dt== "cpu"

    print(dt)

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



# 测试 ones 方法
def test_ones():
    # 测试不同维度的情况
    for shape in [3, (2, 3), (2, 3, 4)]:
        x = bm.ones(shape)
        print(x)
        # 测试 x 的数据类型是否为 ti.Field
        assert isinstance(x, ti.Field)

        # 使用 Taichi 内核验证 x 里的元素是否全为整型 1
        @ti.kernel
        def check_all_ones() -> bool:
            all_ones = True
            for I in ti.grouped(x):
                if x[I] != 1 or ti.cast(x[I], ti.f32) != 1.0:
                    all_ones = False
            return all_ones

        result = check_all_ones()
        assert result == True

# 测试 full 方法
def test_full():
# 测试布尔类型填充
    shape = (3, 3)
    element = True
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.u8
    assert np.all(x.to_numpy() == 1)

    # 测试整数类型填充
    shape = (2, 2)
    element = 5
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32
    assert np.all(x.to_numpy() == 5)

    # 测试浮点数类型填充
    shape = (4,)
    element = 3.14
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64
    assert np.allclose(x.to_numpy(), 3.14)

    # 测试自定义 dtype
    shape = (3, 3)
    element = 10
    custom_dtype = ti.f32
    x = bm.full(shape, element, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f32
    assert np.all(x.to_numpy() == 10)

    # 测试不支持的元素类型
    shape = (2, 2)
    element = "invalid"
    with pytest.raises(TypeError, match="Unsupported fill_value type."):
        bm.full(shape, element)

    # 测试标量 shape
    shape = 5
    element = 1
    x = bm.full(shape, element)
    assert x.shape == (5,)
    assert np.all(x.to_numpy() == 1)

    # 测试空 shape
    shape = ()
    element = 0
    x = bm.full(shape, element)
    assert x.shape == ()
    assert x.to_numpy() == 0

    # 测试多维 shape
    shape = (2, 3, 4)
    element = 7
    x = bm.full(shape, element)
    assert x.shape == (2, 3, 4)
    assert np.all(x.to_numpy() == 7)

if __name__ == '__main__':
    pytest.main(['-q', '-s'])

