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

# 测试 ones_like 方法
def test_ones_like():
    # 测试标量场
    x_scalar = ti.field(dtype=ti.i32, shape=())
    x_scalar[None] = 5
    ones_like_scalar = bm.ones_like(x_scalar)
    assert isinstance(ones_like_scalar, ti.Field)
    assert ones_like_scalar.dtype == ti.i32
    assert ones_like_scalar.shape == ()
    assert ones_like_scalar[None] == 1

    # 测试一维场
    x_1d = ti.field(dtype=ti.f32, shape=(5,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            x_1d[i] = 2.0
    fill_1d()
    ones_like_1d = bm.ones_like(x_1d)
    assert isinstance(ones_like_1d, ti.Field)
    assert ones_like_1d.dtype == ti.i32
    assert ones_like_1d.shape == (5,)
    assert np.all(ones_like_1d.to_numpy() == 1.0)

    # 测试二维场
    x_2d = ti.field(dtype=ti.i64, shape=(3, 4))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            x_2d[i, j] = 10
    fill_2d()
    ones_like_2d = bm.ones_like(x_2d)
    assert isinstance(ones_like_2d, ti.Field)
    assert ones_like_2d.dtype == ti.i32
    assert ones_like_2d.shape == (3, 4)
    assert np.all(ones_like_2d.to_numpy() == 1)

    # 测试不同数据类型
    dtypes = [ti.i8, ti.i16, ti.i32, ti.i64, ti.u8, ti.u16, ti.u32, ti.u64, ti.f32, ti.f64]
    for dtype in dtypes:
        x = ti.field(dtype=dtype, shape=(2,))
        x.fill(100)
        ones_like = bm.ones_like(x)
        assert isinstance(ones_like, ti.Field)
        assert ones_like.dtype == ti.i32
        assert ones_like.shape == (2,)
        assert np.all(ones_like.to_numpy() == 1)

# 测试 full_like 方法
def test_full_like():
    # 测试布尔类型填充
    x_bool = ti.field(dtype=ti.i32, shape=(3, 3))
    element_bool = True
    x = bm.full_like(x_bool, element_bool)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.u8
    assert np.all(x.to_numpy() == 1)

    # 测试整数类型填充
    x_int = ti.field(dtype=ti.f32, shape=(2, 2))
    element_int = 5
    x = bm.full_like(x_int, element_int)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32
    assert np.all(x.to_numpy() == 5)

    # 测试浮点数类型填充
    x_float = ti.field(dtype=ti.i64, shape=(4,))
    element_float = 3.14
    x = bm.full_like(x_float, element_float)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64
    assert np.allclose(x.to_numpy(), 3.14)

    # 测试自定义 dtype
    x_custom = ti.field(dtype=ti.i32, shape=(3, 3))
    element_custom = 10
    custom_dtype = ti.f32
    x = bm.full_like(x_custom, element_custom, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f32
    assert np.all(x.to_numpy() == 10)

    # 测试不支持的元素类型
    x_invalid = ti.field(dtype=ti.i32, shape=(2, 2))
    element_invalid = "invalid"
    with pytest.raises(TypeError, match="Unsupported fill_value type."):
        bm.full_like(x_invalid, element_invalid)

    # 测试多维场
    x_multi = ti.field(dtype=ti.f32, shape=(2, 3, 4))
    element_multi = 7
    x = bm.full_like(x_multi, element_multi)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32
    assert x.shape == (2, 3, 4)
    assert np.all(x.to_numpy() == 7)

# 测试 acosh 方法
def test_acosh():
    # 测试标量输入且值在定义域内的情况
    x_scalar = 2.0
    result_scalar = bm.acosh(x_scalar)
    expected_scalar = np.arccosh(x_scalar)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试标量输入但值不在定义域内的情况
    x_invalid_scalar = 0.5
    with pytest.raises(ValueError, match="must be >= 1.0"):
        bm.acosh(x_invalid_scalar)

    # 测试 ti.Field 输入且所有值都在定义域内的情况
    x_field = ti.field(dtype=ti.f32, shape=(3,))
    x_field.from_numpy(np.array([1.5, 2.0, 3.0], dtype=np.float32))
    result_field = bm.acosh(x_field)
    expected_field = np.arccosh(x_field.to_numpy())
    assert isinstance(result_field, ti.Field)
    assert np.allclose(result_field.to_numpy(), expected_field)

    # 测试 ti.Field 输入但部分值不在定义域内的情况
    x_invalid_field = ti.field(dtype=ti.f32, shape=(3,))
    x_invalid_field.from_numpy(np.array([0.5, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError, match="must be >= 1.0"):
        bm.acosh(x_invalid_field)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a float"):
        bm.acosh(x_invalid_type)

if __name__ == '__main__':
    pytest.main(['-q', '-s'])

