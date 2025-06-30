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
    
# 测试 from_numpy 方法
class TestFromNumpy:
    def test_from_numpy_float32(self):
        # 初始化Taichi（如果尚未初始化）
        ti.init(arch=ti.cpu)  # 使用CPU后端加速测试

<<<<<<< HEAD
# 测试 from_numpy 方法
class TestFromNumpy:
    def test_from_numpy_float32(self):
        # 初始化Taichi（如果尚未初始化）
        ti.init(arch=ti.cpu)  # 使用CPU后端加速测试
=======
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

#测试 to_list 方法   
class TestTolist:
    def test_to_list_empty(self):
        """测试空 Field 是否能正确转换为空列表"""
        field = ti.field(ti.f32, shape=())
        result = bm.to_list(field)
        assert np.allclose(result, [])

    def test_to_list_1d(self):
        """测试 1D Field 是否能正确转换为列表"""
        field = ti.field(ti.i32, shape=(3,))
        field.from_numpy(np.array([1, 2, 3]))
        assert bm.to_list(field) == [1, 2, 3]

    def test_to_list_2d(self):
        """测试 2D Field 是否能正确转换为列表"""
        field = ti.field(ti.f32, shape = (2, 3))
        field.from_numpy(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))

        # 允许浮点数微小误差
        assert np.allclose(bm.to_list(field), [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

#测试 arange 方法
class TestArange():
    def test_arange_simple(self):
        """测试基础范围生成"""
        field = bm.arange(10)
        assert np.allclose(field.to_numpy(), np.arange(10))

    def test_arange_step(self):
        """测试带步长的范围生成"""
        field = bm.arange(0, 10, 2)
        assert np.allclose(field.to_numpy(), np.arange(0,10,2))

    def test_arrange_empty(self):
        """测试空范围生成"""
        field = bm.arange(0)
        assert field is None

    def test_arange_invaild(self):
        """测试无效范围生成"""
        field = bm.arange(-1)
        assert np.allclose(field, np.arange(-1))

#测试 eye 函数
class TestEye:
    def test_eye_square(self):
        """测试对角线为1的方阵"""
        field = bm.eye(1)
        assert np.allclose(field.to_numpy(), np.eye(1))

    def test_eye_rectangle(self):
        """"测试对角线为1的矩形"""
        field = bm.eye(3,4)
        assert np.allclose(field.to_numpy(),np.eye(3,4))

    def test_eye_diagonal_offset(self):
        """测试对角线偏移的方阵"""
        field = bm.eye(3, k=1)
        assert np.allclose(field.to_numpy(), np.eye(3, k=1))

    def test_eye_empty(self):
        """测试空方阵"""
        field = bm.eye(0)
        assert field is None

#测试 Zeros 函数
class TestZeros:
    def test_zeros_1d(self):
        """测试 1D 零矩阵"""
        field = bm.zeros((3,))
        assert np.allclose(field.to_numpy(), np.zeros((3,)))

    def test_zeros_2d(self):
        """测试 2D 零矩阵"""
        field = bm.zeros((2, 3))
        assert np.allclose(field.to_numpy(), np.zeros((2, 3)))

    def test_zeros_empty(self):
        """测试空矩阵"""
        field = bm.zeros((0,))
        assert field is None

#测试 tril 函数
class TestTril:
    def test_tril_square(self):
        """测试下三角方阵"""
        field = bm.tril(3)
        expected = np.array([[1,0,0],
                             [1,1,0],
                             [1,1,1]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_rectangle(self):
        """测试下三角矩阵"""
        field = bm.tril(2,3)
        expected = np.array([[1,0,0],
                             [1,1,0]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_diagonal_offset(self):
        """测试下三角矩阵，偏移对角线"""
        field = bm.tril(3, k=1)
        expected = np.array([[1,1,0],
                             [1,1,1],
                             [1,1,1]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_empty(self):
        """测试空矩阵"""
        field = bm.tril(0)
        assert field is None

#测试 abs 函数
class TestAbs:
    def test_abs_positive(self):
        """测试正数元素的绝对值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_abs_negative(self):    
        """测试负数元素的绝对值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([-1, -2, -3]))        
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.array([1, 2, 3]))

    def test_abs_empty(self):
        """测试空元素的绝对值"""
        field = ti.field(ti.f32, shape=())
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.abs(field.to_numpy()))

    # def test_abs_complex(self):
    #     """测试复数元素的绝对值"""
    #     field = ti.field(ti.c64, shape=(3,))
    #     field.from_numpy(np.array([1+2j, 2-1j, 3+0j]))
    #     result = bm.abs(field)
    #     assert np.allclose(result.to_numpy(), np.array([np.sqrt(5), np.sqrt(5), 3.0]))

#测试 acos 函数
class TestAcos:
    def test_acos_normal_values(self):
        """测试常规值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([0.5, -0.7, 0.9]))
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(np.array([0.5, -0.7, 0.9])))

    def test_acos_boundary_values(self):
        """测试边界值"""
        field = ti.field(ti.f32, shape=(2,))
        field.from_numpy(np.array([1.0, -1.0]))
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(np.array([1.0, -1.0])))

    def test_acos_empty(self):
        """测试空元素"""
        field = ti.field(ti.f32, shape=())
        field[None] = 0.5
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(0.5))

#测试 zeros_like 函数
class TestZerosLike:
    def test_zeros_like_normal(self):
        """测试常规情况"""
        field = ti.field(ti.f32, shape=(2, 3))
        result = bm.zeros_like(field)
        assert np.allclose(result.to_numpy(), np.zeros((2, 3)))

    def test_zeros_like_empty(self):
        """测试空元素"""
        field = ti.field(ti.f32, shape=())
        result = bm.zeros_like(field)
        assert np.allclose(result.to_numpy(), np.zeros_like(field.to_numpy()))

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
    shape = (2,2)
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
    assert ones_like_1d.dtype == ti.f32
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
    assert ones_like_2d.dtype == ti.i64
    assert ones_like_2d.shape == (3, 4)
    assert np.all(ones_like_2d.to_numpy() == 1)

    # 测试不同数据类型
    dtypes = [ti.i8, ti.i16, ti.i32, ti.i64, ti.u8, ti.u16, ti.u32, ti.u64, ti.f32, ti.f64]
    for dtype in dtypes:
        x = ti.field(dtype=dtype, shape=(2,))
        x.fill(100)
        ones_like = bm.ones_like(x)
        assert isinstance(ones_like, ti.Field)
        assert ones_like.dtype == dtype
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

# 测试 asinh 方法
def test_asinh():
    # 测试标量输入情况
    x_scalar = 1.0
    result_scalar = bm.asinh(x_scalar)
    expected_scalar = np.arcsinh(x_scalar)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试标量输入边界值情况
    x_boundary = 0.0
    result_boundary = bm.asinh(x_boundary)
    expected_boundary = np.arcsinh(x_boundary)
    assert np.isclose(result_boundary, expected_boundary)

    # 测试 ti.Field 输入且所有值都在定义域内的情况
    x_field = ti.field(dtype=ti.f32, shape=(3,))
    x_field.from_numpy(np.array([0.5, 1.0, 1.5], dtype=np.float32))
    result_field = bm.asinh(x_field)
    expected_field = np.arcsinh(x_field.to_numpy())
    assert isinstance(result_field, ti.Field)
    assert np.allclose(result_field.to_numpy(), expected_field)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a float"):
        bm.asinh(x_invalid_type)

# 测试 add 方法
def test_add():
    # 测试两个形状相同的 ti.Field 相加成功的情况
    x = ti.field(dtype=ti.f32, shape=(3, 3))
    y = ti.field(dtype=ti.f32, shape=(3, 3))
    # 初始化值
    for i in range(3):
        for j in range(3):
            x[i, j] = i + j
            y[i, j] = (i + j) * 2
    # 执行相加
    result = bm.add(x, y)
    # 验证结果
    for i in range(3):
        for j in range(3):
            assert result[i, j] == x[i, j] + y[i, j]

    # 测试两个形状不同的 ti.Field 相加时抛出 ValueError
    x = ti.field(dtype=ti.f32, shape=(2, 2))
    y = ti.field(dtype=ti.f32, shape=(3, 3))
    with pytest.raises(ValueError, match="Input fields must have the same shape"):
        bm.add(x, y)

    # 测试输入类型不是 ti.Field 时抛出 TypeError
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    with pytest.raises(TypeError, match="Both inputs must be ti.Field"):
        bm.add(x, y)

    # 测试不同数据类型的 ti.Field 相加（如果允许）
    x = ti.field(dtype=ti.f32, shape=(2, 2))
    y = ti.field(dtype=ti.i32, shape=(2, 2))
    # 初始化值
    for i in range(2):
        for j in range(2):
            x[i, j] = 1.5
            y[i, j] = 2
    result = bm.add(x, y)
    # 验证结果是否自动提升类型
    assert result.dtype == ti.f32
    for i in range(2):
        for j in range(2):
            assert result[i, j] == pytest.approx(3.5)
>>>>>>> swh/taichi

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

#测试 to_list 方法   
class TestTolist:
    def test_to_list_empty(self):
        """测试空 Field 是否能正确转换为空列表"""
        field = ti.field(ti.f32, shape=())
        result = bm.to_list(field)
        assert np.allclose(result, [])

    def test_to_list_1d(self):
        """测试 1D Field 是否能正确转换为列表"""
        field = ti.field(ti.i32, shape=(3,))
        field.from_numpy(np.array([1, 2, 3]))
        assert bm.to_list(field) == [1, 2, 3]

    def test_to_list_2d(self):
        """测试 2D Field 是否能正确转换为列表"""
        field = ti.field(ti.f32, shape = (2, 3))
        field.from_numpy(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))

        # 允许浮点数微小误差
        assert np.allclose(bm.to_list(field), [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

#测试 arange 方法
class TestArange():
    def test_arange_simple(self):
        """测试基础范围生成"""
        field = bm.arange(10)
        assert np.allclose(field.to_numpy(), np.arange(10))

    def test_arange_step(self):
        """测试带步长的范围生成"""
        field = bm.arange(0, 10, 2)
        assert np.allclose(field.to_numpy(), np.arange(0,10,2))

    def test_arrange_empty(self):
        """测试空范围生成"""
        field = bm.arange(0)
        assert field is None

    def test_arange_invaild(self):
        """测试无效范围生成"""
        field = bm.arange(-1)
        assert np.allclose(field, np.arange(-1))

#测试 eye 函数
class TestEye:
    def test_eye_square(self):
        """测试对角线为1的方阵"""
        field = bm.eye(1)
        assert np.allclose(field.to_numpy(), np.eye(1))

    def test_eye_rectangle(self):
        """"测试对角线为1的矩形"""
        field = bm.eye(3,4)
        assert np.allclose(field.to_numpy(),np.eye(3,4))

    def test_eye_diagonal_offset(self):
        """测试对角线偏移的方阵"""
        field = bm.eye(3, k=1)
        assert np.allclose(field.to_numpy(), np.eye(3, k=1))

    def test_eye_empty(self):
        """测试空方阵"""
        field = bm.eye(0)
        assert field is None

#测试 Zeros 函数
class TestZeros:
    def test_zeros_1d(self):
        """测试 1D 零矩阵"""
        field = bm.zeros((3,))
        assert np.allclose(field.to_numpy(), np.zeros((3,)))

    def test_zeros_2d(self):
        """测试 2D 零矩阵"""
        field = bm.zeros((2, 3))
        assert np.allclose(field.to_numpy(), np.zeros((2, 3)))

    def test_zeros_empty(self):
        """测试空矩阵"""
        field = bm.zeros((0,))
        assert field is None

#测试 tril 函数
class TestTril:
    def test_tril_square(self):
        """测试下三角方阵"""
        field = bm.tril(3)
        expected = np.array([[1,0,0],
                             [1,1,0],
                             [1,1,1]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_rectangle(self):
        """测试下三角矩阵"""
        field = bm.tril(2,3)
        expected = np.array([[1,0,0],
                             [1,1,0]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_diagonal_offset(self):
        """测试下三角矩阵，偏移对角线"""
        field = bm.tril(3, k=1)
        expected = np.array([[1,1,0],
                             [1,1,1],
                             [1,1,1]])
        assert np.allclose(field.to_numpy(), expected)

    def test_tril_empty(self):
        """测试空矩阵"""
        field = bm.tril(0)
        assert field is None

#测试 abs 函数
class TestAbs:
    def test_abs_positive(self):
        """测试正数元素的绝对值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([1.0, 2.0, 3.0]))
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_abs_negative(self):    
        """测试负数元素的绝对值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([-1, -2, -3]))        
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.array([1, 2, 3]))

    def test_abs_empty(self):
        """测试空元素的绝对值"""
        field = ti.field(ti.f32, shape=())
        result = bm.abs(field)
        assert np.allclose(result.to_numpy(), np.abs(field.to_numpy()))

    # def test_abs_complex(self):
    #     """测试复数元素的绝对值"""
    #     field = ti.field(ti.c64, shape=(3,))
    #     field.from_numpy(np.array([1+2j, 2-1j, 3+0j]))
    #     result = bm.abs(field)
    #     assert np.allclose(result.to_numpy(), np.array([np.sqrt(5), np.sqrt(5), 3.0]))

#测试 acos 函数
class TestAcos:
    def test_acos_normal_values(self):
        """测试常规值"""
        field = ti.field(ti.f32, shape=(3,))
        field.from_numpy(np.array([0.5, -0.7, 0.9]))
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(np.array([0.5, -0.7, 0.9])))

    def test_acos_boundary_values(self):
        """测试边界值"""
        field = ti.field(ti.f32, shape=(2,))
        field.from_numpy(np.array([1.0, -1.0]))
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(np.array([1.0, -1.0])))

    def test_acos_empty(self):
        """测试空元素"""
        field = ti.field(ti.f32, shape=())
        field[None] = 0.5
        result = bm.acos(field)
        assert np.allclose(result.to_numpy(), np.arccos(0.5))

#测试 zeros_like 函数
class TestZerosLike:
    def test_zeros_like_normal(self):
        """测试常规情况"""
        field = ti.field(ti.f32, shape=(2, 3))
        result = bm.zeros_like(field)
        assert np.allclose(result.to_numpy(), np.zeros((2, 3)))

    def test_zeros_like_empty(self):
        """测试空元素"""
        field = ti.field(ti.f32, shape=())
        result = bm.zeros_like(field)
        assert np.allclose(result.to_numpy(), np.zeros_like(field.to_numpy()))
        
if __name__ == '__main__':
    pytest.main(['-q', '-s'])

