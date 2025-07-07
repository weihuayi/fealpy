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
from loguru import logger
from fealpy.backend import backend_manager as bm
import numpy as np

bm.set_backend("taichi")


# 测试 set_default_device 方法
def test_set_default_device():
    # 测试设置为 'cpu'
    bm.set_default_device("cpu")
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
    assert ctx["dtype"] == x.dtype

    # device 应为 'cpu'
    assert ctx["device"] == "cpu"

    print(ctx)


# 测试 device_type 方法
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
    assert dt == "cpu"

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

    @ti.kernel
    def check_bool():
        for i, j in x:
            assert x[i, j] == True

    check_bool()
    # 测试整数类型填充
    shape = (2, 2)
    element = 5
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32

    @ti.kernel
    def check_int():
        for i, j in x:
            assert x[i, j] == 5

    check_int()

    # 测试浮点数类型填充
    shape = (4,)
    element = 3.14
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64

    @ti.kernel
    def check_float():
        for i in x:
            assert x[i] == ti.cast(3.14, x.dtype)

    check_float()

    # 测试自定义 dtype
    shape = (3, 3)
    element = 10
    custom_dtype = ti.f32
    x = bm.full(shape, element, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f32

    @ti.kernel
    def check_custom():
        for i, j in x:
            assert x[i, j] == 10

    check_custom()

    # 测试不支持的元素类型
    shape = (2, 2)
    element = "invalid"
    with pytest.raises(TypeError, match="Unsupported fill_value type."):
        bm.full(shape, element)

    # 测试多维场
    shape = (2, 3, 4)
    element = 7
    x = bm.full(shape, element)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32

    @ti.kernel
    def check_multi():
        for I in ti.grouped(x):
            assert x[I] == 7

    check_multi()


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

    @ti.kernel
    def check_1d():
        for i in ones_like_1d:
            assert ones_like_1d[i] == 1.0

    check_1d()

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

    @ti.kernel
    def check_2d():
        for i, j in ones_like_2d:
            assert ones_like_2d[i, j] == 1

    check_2d()

    # 测试不同数据类型
    dtypes = [
        ti.i8,
        ti.i16,
        ti.i32,
        ti.i64,
        ti.u8,
        ti.u16,
        ti.u32,
        ti.u64,
        ti.f32,
        ti.f64,
    ]
    for dtype in dtypes:
        x = ti.field(dtype=dtype, shape=(2,))

        @ti.kernel
        def fill_x():
            for i in x:
                x[i] = 100

        fill_x()
        ones_like = bm.ones_like(x)
        assert isinstance(ones_like, ti.Field)
        assert ones_like.dtype == dtype
        assert ones_like.shape == (2,)

        @ti.kernel
        def check_dtype():
            for i in ones_like:
                assert ones_like[i] == 1

        check_dtype()


# 测试 full_like 方法
def test_full_like():
    # 测试布尔类型填充
    x_bool = ti.field(dtype=ti.i32, shape=(3, 3))

    @ti.kernel
    def fill_bool():
        for i, j in x_bool:
            x_bool[i, j] = 5

    fill_bool()
    element_bool = True
    x = bm.full_like(x_bool, element_bool)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.u8

    @ti.kernel
    def check_bool():
        for i, j in x:
            assert x[i, j] == True

    check_bool()

    # 测试整数类型填充
    x_int = ti.field(dtype=ti.f32, shape=(2, 2))

    @ti.kernel
    def fill_int():
        for i, j in x_int:
            x_int[i, j] = 2.0

    fill_int()
    element_int = 5
    x = bm.full_like(x_int, element_int)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32

    @ti.kernel
    def check_int():
        for i, j in x:
            assert x[i, j] == 5

    check_int()

    # 测试浮点数类型填充
    x_float = ti.field(dtype=ti.i64, shape=(4,))

    @ti.kernel
    def fill_float():
        for i in x_float:
            x_float[i] = 10

    fill_float()
    element_float = 3.14
    x = bm.full_like(x_float, element_float)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64

    @ti.kernel
    def check_float():
        for i in x:
            assert x[i] == ti.cast(3.14, x.dtype)

    check_float()

    # 测试自定义 dtype
    x_custom = ti.field(dtype=ti.i32, shape=(3, 3))

    @ti.kernel
    def fill_custom():
        for i, j in x_custom:
            x_custom[i, j] = 5

    fill_custom()
    element_custom = 10
    custom_dtype = ti.f32
    x = bm.full_like(x_custom, element_custom, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f32

    @ti.kernel
    def check_custom():
        for i, j in x:
            assert x[i, j] == ti.cast(10, x.dtype)

    check_custom()

    # 测试不支持的元素类型
    x_invalid = ti.field(dtype=ti.i32, shape=(2, 2))

    @ti.kernel
    def fill_invalid():
        for i, j in x_invalid:
            x_invalid[i, j] = 5

    fill_invalid()
    element_invalid = "invalid"
    with pytest.raises(TypeError, match="Unsupported fill_value type."):
        bm.full_like(x_invalid, element_invalid)

    # 测试多维场
    x_multi = ti.field(dtype=ti.f32, shape=(2, 3, 4))

    @ti.kernel
    def fill_multi():
        for I in ti.grouped(x_multi):
            x_multi[I] = 2.0

    fill_multi()
    element_multi = 7
    x = bm.full_like(x_multi, element_multi)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.i32
    assert x.shape == (2, 3, 4)

    @ti.kernel
    def check_multi():
        for I in ti.grouped(x):
            assert x[I] == 7

    check_multi()


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

    @ti.kernel
    def fill_x():
        for i, j in x:
            x[i, j] = i + j

    @ti.kernel
    def fill_y():
        for i, j in y:
            y[i, j] = (i + j) * 2

    fill_x()
    fill_y()
    result = bm.add(x, y)

    @ti.kernel
    def check_result():
        for i, j in result:
            assert result[i, j] == x[i, j] + y[i, j]

    check_result()

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

    @ti.kernel
    def fill_x_float():
        for i, j in x:
            x[i, j] = 1.5

    @ti.kernel
    def fill_y_int():
        for i, j in y:
            y[i, j] = 2

    fill_x_float()
    fill_y_int()
    result = bm.add(x, y)
    assert result.dtype == ti.f32

    @ti.kernel
    def check_mixed_type():
        for i, j in result:
            assert result[i, j] == ti.cast(3.5, result.dtype)

    check_mixed_type()

# 测试 from_numpy 方法  
def test_from_numpy():

    # float型1d
    arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    field = bm.from_numpy(arr)
    assert field.dtype == ti.f32
    assert field.shape == (3,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == arr[i]

    # int型2d
    arr = np.array([[1, 2],[3,4]], dtype=np.int32)
    field = bm.from_numpy(arr)
    assert field.dtype == ti.i32
    assert field.shape == (2, 2)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field[i, j] == arr[i][j]

    # bool型
    arr = np.array([True, False, True], dtype=np.bool)
    field = bm.from_numpy(arr)
    assert field.dtype == ti.u8
    assert field.shape == (3,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == arr[i]

    # 3d
    arr = np.array([[[1, 2, 3], 
                        [4, 5, 6]], 
                    [[7, 8, 9], 
                        [10, 11, 12]]], dtype=np.int32)
    field = bm.from_numpy(arr)
    assert field.shape == (2, 2, 3)
    assert field.dtype == ti.i32
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert field[i, j, k] == arr[i][j][k]

# 测试 tolist 方法   
def test_tolist():

    # 空
    field = ti.field(ti.f32, shape=())
    field[None] = 1.1
    result = bm.tolist(field)
    expected = [1.1]
    # assert field.shape == ()
    # assert isinstance(field, ti.Field)
    # assert field.dtype == ti.f32
    assert np.allclose(result, expected)

    # 数字
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 2.2
    result = bm.tolist(field)
    expected = [2.2]
    assert field.shape == (1,)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f32
    assert np.allclose(result, expected)

    # 1D Field
    field = ti.field(ti.i32, shape=(3,))
    @ti.kernel
    def fill():
        for i in field:
            field[i] = i  + 1
    fill()
    result = bm.tolist(field)
    expected = [1, 2, 3]
    assert field.shape == (3,)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.i32
    assert result == expected

    # 2D Field
    field = ti.field(ti.f32, shape = (2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i  + j  + 1
    fill()
    result = bm.tolist(field)
    expected = [[1.0, 2.0, 3.0], 
                [2.0, 3.0, 4.0]]
    assert field.shape == (2, 3)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f32
    assert result == expected

    # 3D Field
    field = ti.field(ti.f32, shape = (2, 3, 4))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i  + j  + k  + 1
    fill()
    result = bm.tolist(field)
    expected = [[[1.0, 2.0, 3.0, 4.0], 
                [2.0, 3.0, 4.0, 5.0], 
                [3.0, 4.0, 5.0, 6.0]], 
                [[2.0, 3.0, 4.0, 5.0], 
                [3.0, 4.0, 5.0, 6.0], 
                [4.0, 5.0, 6.0, 7.0]]]
    assert field.shape == (2, 3, 4)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f32
    assert result == expected

# 测试 arange 方法
def test_arange():

    # 一个参数
    field = bm.arange(10, dtype=ti.i32)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert field.dtype == ti.i32
    assert field.shape == (10,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == expected[i]

    # 两个参数
    field = bm.arange(0, 10, dtype=ti.i32)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert field.dtype == ti.i32
    assert field.shape == (10,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == expected[i]

    # 三个参数
    field = bm.arange(0, 10, 2,dtype=ti.i32)
    expected = [0, 2, 4, 6, 8]
    assert field.dtype == ti.i32
    assert field.shape == (5,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == expected[i]

    # 单参数为 0
    field = bm.arange(0)
    expected = []
    assert field == expected

    # 单参数为负数
    field = bm.arange(-10)
    expected = []
    assert field == expected

    # 双参数中 N>M
    field = bm.arange(10, 0)
    expected = []
    assert field == expected

    # 三参数中 N>M
    field = bm.arange(10, 0, 2)
    expected = []
    assert field == expected

    # 步长大于总长度            
    field = bm.arange(1, 10, 11, dtype=ti.i32)
    expected = [1]
    assert field.dtype == ti.i32
    assert field.shape == (1,)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        assert field[i] == expected[i]

# 测试 eye 函数
def test_eye(): 

    # 测试空矩阵  
    field = bm.eye(0)
    expected = []
    assert field == expected

    # N = 0
    field = bm.eye(0, 3)
    expected = []
    assert field == expected

    # M = 0
    field = bm.eye(3, 0)
    expected = []
    assert field == expected

    # 一个数字单位阵
    field = bm.eye(1, dtype=ti.i32)
    expcted = [[1]]
    assert field.dtype == ti.i32
    assert field.shape == (1, 1)
    assert isinstance(field, ti.Field)
    assert field[0, 0] == expcted[0][0]                              

    # 测试 3 阶单位阵
    field = bm.eye(3, dtype=ti.i32)
    expected = [[1, 0, 0], 
                [0, 1, 0], 
                [0, 0, 1]]
    assert field.dtype == ti.i32
    assert field.shape == (3, 3)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field[i, j] == expected[i][j]

    # 测试 3 行 4 列单位阵，偏移对角线
    field = bm.eye(3, 4 ,k=1)
    expected = [[0.0, 1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, 0.0, 1.0]]
    assert field.dtype == ti.f64
    assert field.shape == (3, 4)
    assert isinstance(field, ti.Field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field[i, j] == expected[i][j]

# 测试 zeros 函数
def test_zeros():

    # 空矩阵
    field = bm.zeros(0)
    assert len(field) == 0
    assert isinstance(field, ti.Field)
    assert np.array_equal(field, [])

    # 数字零矩阵
    field = bm.zeros(1)
    assert field.shape == (1,)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
    assert field[0] == 0

    # 1d 零矩阵
    field = bm.zeros(3)
    assert field.shape == (3,)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
    assert np.all(field, 0)

    # 2d 零矩阵
    field = bm.zeros((2, 3))
    logger.info(field)
    assert field.shape == (2, 3)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
    assert np.all(field, 0)

    # 3d 零矩阵
    field = bm.zeros((2, 3, 4))
    assert field.shape == (2, 3, 4)
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
    assert np.all(field, 0)


# 测试 tril 函数
def test_tril():

    # 测试数组中只有一个数的情况
    field = ti.field(ti.i32, shape=(1,))
    field[0] = 1
    result = bm.tril(field)
    assert result.shape == (1, 1)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    assert result[0, 0] == 1

    # 2d 下三角方阵
    field = ti.field(ti.i32, shape = (3, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i + j + 1
    fill()
    result = bm.tril(field)
    expected = [[1, 0, 0], 
                [2, 3, 0], 
                [3, 4, 5]]
    assert result.shape == (3, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # 2d 下三角矩阵
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i + j + 1
    fill()
    result = bm.tril(field)
    expected = [[1.0, 0.0, 0.0], 
                [2.0, 3.0, 0.0]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    #  2d 下三角方阵，偏移对角线
    field = ti.field(ti.i32, shape=(3, 3)) 
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i + j + 1
    fill()  
    result = bm.tril(field, k=-1)
    expected = [[0, 0, 0], 
                [2, 0, 0], 
                [3, 4, 0]]
    assert result.shape == (3, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # 3d 下三角方阵
    field = ti.field(ti.i32, shape=(2, 3, 4))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i + j + k + 1
    fill()
    result = bm.tril(field)
    expected = [[[1, 0, 0, 0], 
                 [2, 3, 0, 0], 
                 [3, 4, 5, 0]], 
                [[2, 0, 0, 0], 
                 [3, 4, 0, 0], 
                 [4, 5, 6, 0]]]
    assert result.shape == (2, 3, 4)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 abs 函数
def test_abs():

    # 空
    field = ti.field(ti.f32, shape=())
    field[None] = -1.1
    result = bm.abs(field)
    expected = 1.1
    assert np.allclose(result, expected)

    # int 型
    x = bm.abs(-100)
    assert isinstance(x, int)
    assert x == 100

    # float 型
    x = bm.abs(-1.1)
    assert isinstance(x, float)
    assert np.allclose(x, 1.1)

    # bool 型
    x = bm.abs(False)
    assert x == 0

    y = bm.abs(True)
    assert y == 1

    # field 数字
    field = ti.field(ti.f32, shape=(1,))
    field[0] = -1.1
    result = bm.abs(field)
    expected = 1.1
    assert result.shape == (1,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], expected)

    # 1d
    field = ti.field(ti.i32, shape=(3,))
    field[0] = -1
    field[1] = 2
    field[2] = -3
    result = bm.abs(field)
    expected = [1, 2, 3]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        assert result[i] == expected[i]

    # 2d
    field = ti.field(ti.i32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i - j
    fill()
    result = bm.abs(field)
    expected = [[0, 1, 2], 
                [1, 0, 1]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # 3d
    field = ti.field(ti.i32, shape=(2, 3, 4))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i - j - k
    fill()
    result = bm.abs(field)
    expected = [[[0, 1, 2, 3], 
                    [1, 2, 3, 4], 
                    [2, 3, 4, 5]], 
                [[1, 0, 1, 2], 
                    [0, 1, 2, 3], 
                    [1, 2, 3, 4]]]
    assert result.shape == (2, 3, 4)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.i32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 acos 函数
def test_acos():

    # int 型
    x = bm.acos(0)
    assert np.allclose(x, np.pi/2)

    # float 型
    x = bm.acos(0.5)
    assert np.allclose(x, np.pi/3)

    # bool 型
    x = bm.acos(True)
    assert x == 0

    y = bm.acos(False)
    assert np.allclose(y, np.pi/2)

    # 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.acos(field)
    assert np.allclose(result, np.pi/3)

    # field 数字
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 0.0
    result = bm.acos(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], np.pi/2)

    # 1d field
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0.5
    field[1] = -0.7
    field[2] = 0.9
    result = bm.acos(field)
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], np.arccos(field[i]))

    # 2d field
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i * 0.5 - j * 0.5
    fill()
    result = bm.acos(field)
    expected = [[np.pi/2, np.pi*2/3, np.pi], 
                [np.pi/3, np.pi/2, np.pi*2/3]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # 3d field
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i / 2 - j / 2 - k / 2
    fill()
    result = bm.acos(field)
    expected = [[[np.pi/2, np.pi*2/3], 
                    [np.pi*2/3, np.pi]], 
                [[np.pi/3, np.pi/2], 
                    [np.pi/2, np.pi*2/3]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])

# 测试 zeros_like 函数
def test_zeros():

    # 空
    field = ti.field(ti.f32, shape=())
    result = bm.zeros_like(field)
    assert result.shape == ()
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert result[None] == 0

    # 数字
    field = ti.field(ti.f32, shape=(1,))
    result = bm.zeros_like(field)
    assert result.shape == (1,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert result[0] == 0

    # 1d
    field = ti.field(ti.f32, shape=(3,))
    result = bm.zeros_like(field)
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert np.all(result, 0)

    # 2d
    field = ti.field(ti.f32, shape=(2, 3))
    result = bm.zeros_like(field)
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert np.all(result, 0)

    # 3d
    field = ti.field(ti.f32, shape=(2, 3, 4))
    result = bm.zeros_like(field)
    assert result.shape == (2, 3, 4)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert np.all(result, 0)

# 测试 asin 函数
def test_asin():

    # int 型
    x = bm.asin(0)
    assert x == 0

    # float 型
    x = bm.asin(0.5)
    assert np.allclose(x, np.pi/6)

    # bool 型
    x = bm.asin(True)
    assert np.allclose(x, np.pi/2)

    y = bm.asin(False)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.asin(field)
    assert np.allclose(result, np.pi/6)

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = -0.5
    result = bm.asin(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], -np.pi/6)

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0.5
    field[1] = -0.7
    field[2] = 0.9
    result = bm.asin(field)
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], np.arcsin(field[i]))

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i * 0.5 - j * 0.5
    fill()
    result = bm.asin(field)
    expected = [[0, -np.pi/6, -np.pi/2], 
                [np.pi/6, 0, -np.pi/6]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i / 2 - j / 2 - k / 2
    fill()
    result = bm.asin(field)
    expected = [[[0, -np.pi/6], 
                 [-np.pi/6, -np.pi/2]], 
                [[np.pi/6, 0], 
                 [0, -np.pi/6]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])

# 测试 atan 函数
def test_atan():

    # int 型
    x = bm.atan(1)
    assert x == np.pi/4

    # float 型
    x = bm.atan(0.5)
    assert np.allclose(x, np.atan(0.5))

    # bool 型
    x = bm.atan(True)
    assert x == np.pi/4

    y = bm.atan(False)
    assert y ==0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.atan(field)
    assert np.allclose(result, np.atan(0.5))

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 0.5
    result = bm.atan(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], np.atan(0.5))

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0.5
    field[1] = -0.7
    field[2] = 0.9
    result = bm.atan(field)
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], np.arctan(field[i]))

    # field 2d
    field = ti.field(ti.f32, shape = (2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i * 0.5 - j * 0.5
    fill()
    result = bm.atan(field)
    expected = [[0, np.arctan(-0.5), -np.pi/4], 
                [np.arctan(0.5), 0, np.arctan(-0.5)]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i / 2 - j / 2 - k / 2
    fill()
    result = bm.atan(field)
    expected = [[[0, np.arctan(-0.5)], 
                 [np.arctan(-0.5), -np.pi/4]], 
                [[np.arctan(0.5), 0], 
                 [0, np.arctan(-0.5)]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])
                
# 测试 atan2 函数
def test_atan2():

    # int 型
    x = bm.atan2(1, 1)
    assert x == np.pi/4

    # float 型
    x = bm.atan2(0.5, 1)
    assert np.allclose(x, np.arctan(0.5))

    # bool 型
    x = bm.atan2(True, 1)
    assert x == np.pi/4

    y = bm.atan2(False, 0.5)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    Field = ti.field(ti.f32, shape=())
    Field[None] = 1.0
    result = bm.atan2(field, Field)
    assert np.allclose(result, np.arctan(0.5))

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 0.5
    Field = ti.field(ti.f32, shape=(1,))
    Field[0] = 1.0
    result = bm.atan2(field, Field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], np.arctan(0.5))

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0.5
    field[1] = -0.7
    field[2] = 0.9
    Field = ti.field(ti.f32, shape=(3,))
    Field[0] = 1.0
    Field[1] = 1.0
    Field[2] = 1.0
    result = bm.atan2(field, Field)
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], np.arctan2(field[i], Field[i]))

    # field 2d
    field = ti.field(ti.f32, shape = (2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i * 0.5 - j * 0.5
    fill()
    Field = ti.field(ti.f32, shape = (2, 3))
    @ti.kernel
    def fill():
        for i, j in Field:
            Field[i, j] = i * 0.5 + j * 0.5 + 1
    fill()
    result = bm.atan2(field, Field)
    expected = [[0, np.arctan2(-0.5, 1.5), np.arctan2(-1, 2)], 
                [np.arctan2(0.5, 1.5), 0, np.arctan2(-0.5, 2.5)]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = i / 2 - j / 2 - k / 2
    fill()
    Field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in Field:
            Field[i, j, k] = i / 2 + j / 2 + k / 2 + 1
    fill()
    result = bm.atan2(field, Field)
    expected = [[[0, np.arctan2(-0.5, 1.5)], 
                 [np.arctan2(-0.5, 1.5), np.arctan2(-1, 2)]], 
                [[np.arctan2(0.5, 1.5), 0], 
                 [0, np.arctan2(-0.5, 2.5)]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])

# 测试 ceil 函数
def test_ceil():

    # int 型
    x = bm.ceil(0)
    assert x == 0

    # float 型
    x = bm.ceil(0.5)
    assert x == 1

    y = bm.ceil(-1.5)
    assert y == -1

    # bool 型
    x = bm.ceil(True)
    assert x == 1

    y = bm.ceil(False)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.ceil(field)
    assert field.shape == ()
    assert field.dtype == ti.f32
    assert result == 1

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = -0.5
    result = bm.ceil(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == 0

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 3.5
    field[1] = -6.6
    field[2] = 2.5
    result = bm.ceil(field)
    expected = [4, -6, 3]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert result[i] == expected[i]

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * 0.5
    fill()
    result = bm.ceil(field)
    expected = [[0, 0, 1], 
                [0, 1, -1]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * 0.5
    fill()
    result = bm.ceil(field)
    expected = [[[0, 0], 
                 [0, 1]], 
                [[0, 1], 
                 [1, -1]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 clip 函数
def test_clip():

    # int 型（传入的参数在 min 和 max 之间）
    x = bm.clip(2, 0, 5)
    assert x == 2

    # float 型（传入的参数大于 max）
    x = bm.clip(10.0, 0.0, 5.0)
    assert x == 5.0

    # bool 型 （传入的参数小于 min）
    x = bm.clip(False, 2, 7)
    assert x == 2

    # a_min 为 None
    x = bm.clip(2, a_max = 5)
    assert x == 2

    # a_max 为 None
    x = bm.clip(2, a_min = 4)
    assert x == 4

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 2.5
    result = bm.clip(field, 0.0, 5.0)
    assert result == 2.5

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 2.5
    result = bm.clip(field, 0.0, 5.0)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == 2.5

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 2.5
    field[1] = -0.7
    field[2] = 10.0
    result = bm.clip(field, 0.0, 5.0)
    expected = [2.5, 0.0, 5.0]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert result[i] == expected[i]

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j)
    fill()
    result = bm.clip(field, -2.0, 2.0)
    expected = [[0.0, -1.0, 2.0], 
                [-1.0, 2.0, -2.0]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k)
    fill()
    result = bm.clip(field, -2.0, 2.0)
    expected = [[[0.0, -1.0], 
                 [-1.0, 2.0]], 
                [[-1.0, 2.0], 
                 [2.0, -2.0]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 cos 函数
def test_cos():

    # int 型
    x = bm.cos(0)
    assert x == 1

    # float 型
    x = bm.cos(np.pi/3)
    assert np.allclose(x, 0.5)

    # bool 型
    x = bm.cos(True)
    assert x == np.cos(1)

    y = bm.cos(False)
    assert y == 1

    # field 空
    field = ti.field(ti.f32, shape = ())
    field[None] = np.pi/3
    result = bm.cos(field)
    assert np.allclose(result, 0.5)

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = np.pi*2/3
    result = bm.cos(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], -0.5)

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = np.pi/2
    field[1] = np.pi/4
    field[2] = np.pi*2/3
    result = bm.cos(field)
    expected = [0, np.sqrt(2)/2, -0.5]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], expected[i], atol = 1e-6)

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * np.pi/6
    fill()
    result = bm.cos(field)
    expected = [[1, np.sqrt(3)/2, 0.5], 
                [np.sqrt(3)/2, 0.5, 0]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j], atol = 1e-6)

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * np.pi/6
    fill()
    result = bm.cos(field)
    expected = [[[1, np.sqrt(3)/2], 
                 [np.sqrt(3)/2, 0.5]], 
                [[np.sqrt(3)/2, 0.5], 
                 [0.5, 0]]]

    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k], atol = 1e-6)

# 测试 cosh 函数
def test_cosh():

    # int 型
    x = bm.cosh(0)
    assert x == 1

    # float 型
    x = bm.cosh(0.5)
    assert x == (np.exp(0.5) + np.exp(-0.5))/2

    # bool 型
    x = bm.cosh(True)
    assert x == (np.exp(1) + np.exp(-1))/2

    y = bm.cosh(False)
    assert y == 1

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.cosh(field)
    assert np.allclose(result, (np.exp(0.5) + np.exp(-0.5))/2)

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 0.5
    result = bm.cosh(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], (np.exp(0.5) + np.exp(-0.5))/2)

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0
    field[1] = 0.5
    field[2] = True
    result = bm.cosh(field)
    expected = [1, (np.exp(0.5) + np.exp(-0.5))/2, (np.exp(1) + np.exp(-1))/2]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], expected[i])

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * 0.5
    fill()
    result = bm.cosh(field)
    expected = [[1, (np.exp(-0.5) + np.exp(0.5))/2, (np.exp(1) + np.exp(-1))/2], 
                [(np.exp(-0.5) + np.exp(0.5))/2, (np.exp(1) + np.exp(-1))/2, (np.exp(-1.5) + np.exp(1.5))/2]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * 0.5
    fill()
    result = bm.cosh(field)
    expected = [[[1, (np.exp(-0.5) + np.exp(0.5))/2], 
                 [(np.exp(-0.5) + np.exp(0.5))/2, (np.exp(1) + np.exp(-1))/2]], 
                [[(np.exp(-0.5) + np.exp(0.5))/2, (np.exp(1) + np.exp(-1))/2], 
                 [(np.exp(1) + np.exp(-1))/2, (np.exp(-1.5) + np.exp(1.5))/2]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])

# 测试 floor 函数
def test_floor():

    # int 型
    x = bm.floor(1)
    assert x == 1

    # float 型
    x = bm.floor(1.5)
    assert x == 1

    # bool 型
    x = bm.floor(True)
    assert x == 1

    y = bm.floor(False)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = -1.5
    result = bm.floor(field)
    assert result == -2

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 7.8
    result = bm.floor(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == 7

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0
    field[1] = -6.6
    field[2] = True
    result = bm.floor(field)
    expected = [0, -7, 1]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert result[i] == expected[i]

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * 1.7
    fill()
    result = bm.floor(field)
    expected = [[0, -2, 3], 
                [-2, 3, -6]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * 1.7
    fill()
    result = bm.floor(field)
    expected = [[[0, -2], 
                 [-2, 3]], 
                [[-2, 3], 
                 [3, -6]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 floor_divide 函数
def test_floor_divide():

    # int 型
    x = bm.floor_divide(10, 3)
    assert x == 3

    # float 型
    x = bm.floor_divide(-10.5, 2.4)
    assert x == -5

    # bool 型
    x = bm.floor_divide(True, 0.7)
    assert x == 1

    y = bm.floor_divide(False, 12.4)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 6.6
    Field = ti.field(ti.f32, shape=())
    Field[None] = 2
    result = bm.floor_divide(field, Field)
    assert result == 3

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = -8.8
    Field = ti.field(ti.f32, shape=(1,))
    Field[0] = 2.4
    result = bm.floor_divide(field, Field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == -4

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 10
    field[1] = -12.6
    field[2] = True
    Field = ti.field(ti.f32, shape=(3,))
    Field[0] = 2
    Field[1] = 3
    Field[2] = 2
    result = bm.floor_divide(field, Field)
    expected = [5, -5, 0]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert result[i] == expected[i]

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i + j * 2
    fill()
    Field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in Field:
            Field[i, j] = i - j * 0.5 + 2
    fill()
    result = bm.floor_divide(field, Field)
    expected = [[0, 1, 4], 
                [0, 1, 2]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result[i, j] == expected[i][j]

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k)
    fill()
    Field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in Field:
            Field[i, j, k] = (-1) ** (i+j+k) + 0.1
    fill()
    result = bm.floor_divide(field, Field)
    expected = [[[0, 1], 
                 [1, 1]], 
                [[1, 1], 
                 [1, 3]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result[i, j, k] == expected[i][j][k]

# 测试 sin 函数
def test_sin():

    # int 型
    x = bm.sin(0)
    assert x == 0

    # float 型
    x = bm.sin(np.pi/2)
    assert x == 1

    # bool 型
    x = bm.sin(True)
    assert x == np.sin(1)

    y = bm.sin(False)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = np.pi/2
    result = bm.sin(field)
    assert result == 1

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = np.pi/6
    result = bm.sin(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == 0.5

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0
    field[1] = np.pi/2
    field[2] = True
    result = bm.sin(field)
    expected = [0, 1, np.sin(1)]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], expected[i])

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * np.pi/6
    fill()
    result = bm.sin(field)
    expected = [[0, -0.5, np.sqrt(3)/2], 
                [-0.5, np.sqrt(3)/2, -1]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * np.pi/6
    fill()
    result = bm.sin(field)
    expected = [[[0, -0.5], 
                 [-0.5, np.sqrt(3)/2]], 
                [[-0.5, np.sqrt(3)/2], 
                 [np.sqrt(3)/2, -1]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])
                
# 测试 sinh 函数
def test_sinh():

    # int 型
    x = bm.sinh(0)
    assert x == 0

    # float 型
    x = bm.sinh(0.5)
    assert x == (np.exp(0.5) - np.exp(-0.5))/2

    # bool 型
    x = bm.sinh(True)
    assert x == (np.exp(1) - np.exp(-1))/2

    y = bm.sinh(False)
    assert y == 0

    # field 空
    field = ti.field(ti.f32, shape=())
    field[None] = 0.5
    result = bm.sinh(field)
    assert np.allclose(result, (np.exp(0.5) - np.exp(-0.5))/2)

    # field 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 0.5
    result = bm.sinh(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert np.allclose(result[0], (np.exp(0.5) - np.exp(-0.5))/2)

    # field 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 0
    field[1] = 0.5
    field[2] = True
    result = bm.sinh(field)
    expected = [0, (np.exp(0.5) - np.exp(-0.5))/2, (np.exp(1) - np.exp(-1))/2]
    assert result.shape == (3,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        assert np.allclose(result[i], expected[i])

    # field 2d
    field = ti.field(ti.f32, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = (-1) ** (i+j) * (i+j) * 0.5
    fill()
    result = bm.sinh(field)
    expected = [[0, (np.exp(-0.5) - np.exp(0.5))/2, (np.exp(1) - np.exp(-1))/2], 
                [(np.exp(-0.5) - np.exp(0.5))/2, (np.exp(1) - np.exp(-1))/2, (np.exp(-1.5) - np.exp(1.5))/2]]
    assert result.shape == (2, 3)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert np.allclose(result[i, j], expected[i][j])

    # field 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    @ti.kernel
    def fill():
        for i, j, k in field:
            field[i, j, k] = (-1) ** (i+j+k) * (i+j+k) * 0.5
    fill()
    result = bm.sinh(field)
    expected = [[[0, (np.exp(-0.5) - np.exp(0.5))/2], 
                 [(np.exp(-0.5) - np.exp(0.5))/2, (np.exp(1) - np.exp(-1))/2]], 
                [[(np.exp(-0.5) - np.exp(0.5))/2, (np.exp(1) - np.exp(-1))/2], 
                 [(np.exp(1) - np.exp(-1))/2, (np.exp(-1.5) - np.exp(1.5))/2]]]
    assert result.shape == (2, 2, 2)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert np.allclose(result[i, j, k], expected[i][j][k])

# 测试 trace 函数
def test_trace():

    # 2d field(三阶方阵)
    field = ti.field(ti.f32, shape=(3, 3))
    field.fill(1)
    field[1, 1] = 2
    result = bm.trace(field)
    assert result == 4

    # 2d field(一阶方阵)
    field = ti.field(ti.f32, shape=(1, 1))
    field[0, 0] = 2
    result = bm.trace(field)
    assert result == 2

# 测试 insert 函数
def test_insert():

    pass

# 测试 unique 函数
def test_unique():

    # 空
    field = ti.field(ti.f32, shape=())
    field[None] = 1
    result = bm.unique(field)
    assert result.shape == (1,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    assert result[0] == 1

    # 数字型
    field = ti.field(ti.f32, shape=(1,))
    field[0] = 1
    result = bm.unique(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f32
    assert result[0] == 1

    # 1d
    field = ti.field(ti.f32, shape=(3,))
    field[0] = 1
    field[1] = 2
    field[2] = 1
    result = bm.unique(field)
    excepted = [1, 2]
    assert result.shape == (2,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f32
    for i in range(result.shape[0]):
        assert result[i] == excepted[i]

    # 2d
    field = ti.field(ti.f32, shape=(2, 3))
    field[0, 0] = 1
    field[0, 1] = 2
    field[0, 2] = 3
    field[1, 0] = 2
    field[1, 1] = 1
    field[1, 2] = 3
    result = bm.unique(field)
    excepted = [1, 2, 3]
    assert isinstance(result, ti.Field)
    assert result.shape == (3,)
    assert result.dtype == ti.f32
    for i in range(result.shape[0]):
        assert result[i] == excepted[i]

    # 3d
    field = ti.field(ti.f32, shape=(2, 2, 2))
    field[0, 0, 0] = 1
    field[0, 0, 1] = 2
    field[0, 1, 0] = 3
    field[0, 1, 1] = 2
    field[1, 0, 0] = 2
    field[1, 0, 1] = 1
    field[1, 1, 0] = 3
    field[1, 1, 1] = 2
    result = bm.unique(field)
    excepted = [1, 2, 3]
    assert isinstance(result, ti.Field)
    assert result.shape == (3,)
    assert result.dtype == ti.f32
    for i in range(result.shape[0]):
        assert result[i] == excepted[i]

if __name__ == "__main__":
    pytest.main(["-q", "-s"])

