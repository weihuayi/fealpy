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
import taichi.math as tm
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

    x = ti.field(dtype=ti.f64, shape=(2, 3))

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
    x = ti.field(dtype=ti.f64, shape=(2, 3))

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
    x = ti.field(dtype=ti.f64, shape=(2, 3))

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
                if x[I] != 1 or ti.cast(x[I], ti.f64) != 1.0:
                    all_ones = False
            return all_ones

        result = check_all_ones()
        assert result == True

    # 测试无效形状类型
    with pytest.raises(ValueError, match="Shape must be an int or a Tuple\[int, ...\]."):
        bm.ones("invalid_shape")

    with pytest.raises(ValueError, match="Shape must be an int or a Tuple\[int, ...\]."):
        bm.ones((1, "2"))

    # 测试零形状
    with pytest.raises(ValueError, match="Shape dimensions must be greater than 0."):
        bm.ones((0,))


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
    custom_dtype = ti.f64
    x = bm.full(shape, element, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64

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
    x_1d = ti.field(dtype=ti.f64, shape=(5,))

    @ti.kernel
    def fill_1d():
        for i in x_1d:
            x_1d[i] = 2.0

    fill_1d()
    ones_like_1d = bm.ones_like(x_1d)
    assert isinstance(ones_like_1d, ti.Field)
    assert ones_like_1d.dtype == ti.f64
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
    x_int = ti.field(dtype=ti.f64, shape=(2, 2))

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
    custom_dtype = ti.f64
    x = bm.full_like(x_custom, element_custom, dtype=custom_dtype)
    assert isinstance(x, ti.Field)
    assert x.dtype == ti.f64

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
    x_multi = ti.field(dtype=ti.f64, shape=(2, 3, 4))
    
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

    # 测试 ti.Field 输入且所有值都在定义域内的情况
    x_field = ti.field(dtype=ti.f64, shape=(3,))
    x_field.from_numpy(np.array([1.5, 2.0, 3.0], dtype=np.float32))
    result_field = bm.acosh(x_field)
    expected_field = np.arccosh(x_field.to_numpy())
    assert isinstance(result_field, ti.Field)
    assert np.allclose(result_field.to_numpy(), expected_field)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a scalar"):
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
    x_field = ti.field(dtype=ti.f64, shape=(3,))
    x_field.from_numpy(np.array([0.5, 1.0, 1.5], dtype=np.float32))
    result_field = bm.asinh(x_field)
    expected_field = np.arcsinh(x_field.to_numpy())
    assert isinstance(result_field, ti.Field)
    assert np.allclose(result_field.to_numpy(), expected_field)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a scalar"):
        bm.asinh(x_invalid_type)


# 测试 add 方法
def test_add():
    # 测试两个形状相同的 ti.Field 相加成功的情况
    x = ti.field(dtype=ti.f64, shape=(3, 3))
    y = ti.field(dtype=ti.f64, shape=(3, 3))

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
    x = ti.field(dtype=ti.f64, shape=(2, 2))
    y = ti.field(dtype=ti.f64, shape=(3, 3))
    with pytest.raises(ValueError, match="Input fields must have the same shape"):
        bm.add(x, y)

    # 测试输入类型不是 ti.Field 时抛出 TypeError
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    with pytest.raises(TypeError, match="Both inputs must be ti.Field"):
        bm.add(x, y)

    # 测试不同数据类型的 ti.Field 相加（如果允许）
    x = ti.field(dtype=ti.f64, shape=(2, 2))
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
    assert result.dtype == ti.f64

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
    for i in range(field.shape[0]):
        assert field.dtype == ti.f32
        assert field.shape == (3,)
        assert isinstance(field, ti.Field)

        assert field[i] == arr[i]

    # int型2d
    arr = np.array([[1, 2],[3,4]], dtype=np.int32)
    field = bm.from_numpy(arr)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field.dtype == ti.i32
            assert field.shape == (2, 2)
            assert isinstance(field, ti.Field)
            assert field[i, j] == arr[i][j]

    # bool型
    arr = np.array([True, False, True], dtype=np.bool)
    field = bm.from_numpy(arr)
    for i in range(field.shape[0]):
        assert field.dtype == ti.u8
        assert field.shape == (3,)
        assert isinstance(field, ti.Field)
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
    field = ti.field(ti.f64, shape=())
    field[None] = 1.1
    result = bm.tolist(field)
    expected = [1.1]
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
    assert np.allclose(result, expected)

    # 数字
    field = ti.field(ti.f64, shape=(1,))
    field[0] = 2.2
    result = bm.tolist(field)
    expected = [2.2]
    assert isinstance(field, ti.Field)
    assert field.dtype == ti.f64
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
    field = ti.field(ti.f64, shape = (2, 3))

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
    assert field.dtype == ti.f64
    assert result == expected

    # 3D Field
    field = ti.field(ti.f64, shape = (2, 3, 4))
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
    assert field.dtype == ti.f64
    assert result == expected

# 测试 arange 方法
def test_arange():

    # 一个参数
    field = bm.arange(10, dtype=ti.i32)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(field.shape[0]):
        assert field.dtype == ti.i32
        assert isinstance(field, ti.Field)
        assert field[i] == expected[i]

    # 两个参数
    field = bm.arange(0, 10, dtype=ti.i32)
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(field.shape[0]):
        assert field.dtype == ti.i32
        assert isinstance(field, ti.Field)
        assert field[i] == expected[i]

    # 三个参数
    field = bm.arange(0, 10, 2,dtype=ti.i32)
    expected = [0, 2, 4, 6, 8]
    for i in range(field.shape[0]):
        assert field.dtype == ti.i32
        assert isinstance(field, ti.Field)
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
    for i in range(field.shape[0]):
        assert field.dtype == ti.i32
        assert isinstance(field, ti.Field)
        assert field[i] == expected[i]

    # 无参数
    with pytest.raises(ValueError, match="arange\(\) requires stop to be specified."):
        bm.arange(None)

    # 参数大于 3
    with pytest.raises(ValueError, match="arange expects 1~3 arguments \(stop \| start, stop \| start, stop, step\)"):
        bm.arange(1,3,5,7,9)

    # 步长为 0
    with pytest.raises(ValueError, match="step must not be zero"):
        bm.arange(1,3,0)


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
    assert field[0, 0] == expcted[0][0]                              

    # 测试 3 阶单位阵
    field = bm.eye(3, dtype=ti.i32)
    expected = [[1, 0, 0], 
                [0, 1, 0], 
                [0, 0, 1]]
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field.dtype == ti.i32
            assert field[i, j] == expected[i][j]

    # 测试 3 行 4 列单位阵，偏移对角线
    field = bm.eye(3, 4 ,k=1)
    expected = [[0.0, 1.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, 0.0, 1.0]]
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert field.dtype == ti.f64
            assert field[i, j] == expected[i][j]

    # N 为 None
    with pytest.raises(ValueError, match="Both N and M are None. At least one dimension must be specified for eye()."):
        bm.eye(None)

    # N 为负数
    with pytest.raises(ValueError, match="N and M must be positive integers, got N=-1, M=3"):
        bm.eye(-1,3)

    # N 不是 int 型
    with pytest.raises(TypeError, match="N must be an integer, got 1.2."):
        bm.eye(1.2)


# 测试 zeros 函数
def test_zeros():

    # 空矩阵
    field = bm.zeros(0)
    assert len(field) == 0
    assert np.array_equal(field, [])

    # 数字零矩阵
    field = bm.zeros(1)
    assert field.shape == (1,)
    assert field.dtype == ti.f64
    assert field[0] == 0

    # 1d 零矩阵
    field = bm.zeros(3)
    assert field.shape == (3,)
    assert field.dtype == ti.f64
    assert np.all(field, 0)

    # 2d 零矩阵
    field = bm.zeros((2, 3))
    logger.info(field)
    assert field.shape == (2, 3)
    assert field.dtype == ti.f64
    assert np.all(field, 0)

    # 3d 零矩阵
    field = bm.zeros((2, 3, 4))
    assert field.shape == (2, 3, 4)
    assert field.dtype == ti.f64
    assert np.all(field, 0)

    #shape为-1
    with pytest.raises(ValueError, match="Shape must be a non-negative integer, got (-1)."):
        bm.zeros((-1))


# 测试 tril 函数
def test_tril():

    # 测试数组中只有一个数的情况
    field = ti.field(ti.i32, shape=(1,))
    field[0] = 1
    result = bm.tril(field)
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
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result.shape == (3, 3)
            assert isinstance(result, ti.Field)
            assert result.dtype == ti.i32
            assert result[i, j] == expected[i][j]

    # 2d 下三角矩阵
    field = ti.field(ti.f64, shape=(2, 3))
    @ti.kernel
    def fill():
        for i, j in field:
            field[i, j] = i + j + 1
    fill()
    result = bm.tril(field)
    expected = [[1.0, 0.0, 0.0], 
                [2.0, 3.0, 0.0]]
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result.shape == (2, 3)
            assert isinstance(result, ti.Field)
            assert result.dtype == ti.f64
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
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result.shape == (3, 3)
            assert isinstance(result, ti.Field)
            assert result.dtype == ti.i32
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
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result.shape == (2, 3, 4)
                assert isinstance(result, ti.Field)
                assert result.dtype == ti.i32
                assert result[i, j, k] == expected[i][j][k]

    # 空
    field = ti.field(ti.f64, shape=())
    with pytest.raises(ValueError, match="Input field is a scalar \(0D\)\, tril is not defined for scalars\."):
        bm.tril(field)

    # field 为 None
    with pytest.raises(ValueError, match="Input field is None. Please provide a valid Taichi field."):
        bm.tril(None)


# 测试 abs 函数
def test_abs():

    # 空
    field = ti.field(ti.f64, shape=())
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
    field = ti.field(ti.f64, shape=(1,))
    field[0] = -1.1
    result = bm.abs(field)
    expected = 1.1
    assert result.shape == (1,)
    assert isinstance(result, ti.Field)
    assert result.dtype == ti.f64
    assert np.allclose(result[0], expected)

    # 1d
    field = ti.field(ti.i32, shape=(3,))
    field[0] = -1
    field[1] = 2
    field[2] = -3
    result = bm.abs(field)
    expected = [1, 2, 3]
    for i in range(field.shape[0]):
        assert result.shape == (3,)
        assert isinstance(result, ti.Field)
        assert result.dtype == ti.i32
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
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            assert result.shape == (2, 3)
            assert isinstance(result, ti.Field)
            assert result.dtype == ti.i32
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
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for k in range(field.shape[2]):
                assert result.shape == (2, 3, 4)
                assert isinstance(result, ti.Field)
                assert result.dtype == ti.i32
                assert result[i, j, k] == expected[i][j][k]

    # 无参数
    with pytest.raises(TypeError, match="Unsupported type for abs: <class 'NoneType'>. Expected int, float, bool, or ti.Field."):
        bm.abs(None)

#测试 acos 函数
    def test_acos():
        
        # int 型
        x = bm.acos(0)
        assert np.allclose(x, np.pi/2)
        
        # float 型
        x = bm.acos(0.5)
        assert np.allclose(x, np.pi/3)

        # bool 型
        x = bm.acos(True)
        assert np.allclose(x, 0)

        # 空
        field = ti.field(ti.f64, shape=())
        field[None] = 0.5
        result = bm.acos(field)
        assert result.shape == ()
        assert result.dtype == ti.f64
        assert np.allclose(result[None], np.pi/3)

        # field 数字
        field = ti.field(ti.f64, shape=(1,))
        field[0] = 0.0
        result = bm.acos(field)
        assert result.shape == (1,)
        assert result.dtype == ti.f64
        assert np.allclose(result[0], np.pi/2)

        # 1d field
        field = ti.field(ti.f64, shape=(3,))
        field[0] = 0.5
        field[1] = -0.7
        field[2] = 0.9
        result = bm.acos(field)
        for i in range(field.shape[0]):
            assert result.shape == (3,)
            assert isinstance(result, ti.Field)
            assert result.dtype == ti.f64
            assert np.allclose(result[i], np.arccos(field[i]))

        # 2d field
        field = ti.field(ti.f64, shape=(2, 3))
        @ti.kernel
        def fill():
            for i, j in field:
                field[i, j] = i * 0.5 - j * 0.5
        fill()
        result = bm.acos(field)
        expected = [[np.pi/2, np.pi*2/3, np.pi], 
                    [np.pi/3, np.pi/2, np.pi*2/3]]
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                assert result.shape == (2, 3)
                assert isinstance(result, ti.Field)
                assert result.dtype == ti.f64
                assert np.allclose(result[i, j], expected[i][j])

        # 3d field
        field = ti.field(ti.f64, shape=(2, 2, 2))
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
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                for k in range(field.shape[2]):
                    assert result.shape == (2, 2, 2)
                    assert isinstance(result, ti.Field)
                    assert result.dtype == ti.f64
                    assert np.allclose(result[i, j, k], expected[i][j][k])

        # 无参数
        with pytest.raises(ValueError, match="Input field is None. Please provide a valid Taichi field."):
            bm.acos(None)

        # 参数为一维空数组
        field = ti.field(ti.f64, shape=(1,))
        with pytest.raises(ValueError, match="ti\.field shape \(1,\) does not match the numpy array shape \(0,\)"):
            result = field.from_numpy(np.array([]))
            bm.acos(result)

#测试 zeros_like 函数
def test_zeros():

    # 空
    field = ti.field(ti.f64, shape=())
    result = bm.zeros_like(field)
    assert result.shape == ()
    assert result.dtype == ti.f64
    assert result[None] == 0

    # 数字
    field = ti.field(ti.f64, shape=(1,))
    result = bm.zeros_like(field)
    assert result.shape == (1,)
    assert result.dtype == ti.f64
    assert result[0] == 0

    # 1d
    field = ti.field(ti.f64, shape=(3,))
    result = bm.zeros_like(field)
    assert result.shape == (3,)
    assert result.dtype == ti.f64
    assert np.all(result, 0)

    # 2d
    field = ti.field(ti.f64, shape=(2, 3))
    result = bm.zeros_like(field)
    assert result.shape == (2, 3)
    assert result.dtype == ti.f64
    assert np.all(result, 0)

    # 3d
    field = ti.field(ti.f64, shape=(2, 3, 4))
    result = bm.zeros_like(field)
    assert result.shape == (2, 3, 4)
    assert result.dtype == ti.f64
    assert np.all(result, 0)

    # 无参数
    with pytest.raises(ValueError, match="Input field is None. Please provide a valid Taichi field."):
        bm.zeros_like(None)


# 测试 atanh 方法
def test_atanh():
    # 测试 0 维标量输入情况 （float 类型）
    x_scalar_f = 0.5
    result_scalar = bm.atanh(x_scalar_f)
    expected_scalar = np.arctanh(x_scalar_f)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试 0 维标量输入情况 （int 类型）
    x_scalar_int = 0
    result_scalar_int = bm.atanh(x_scalar_int)
    expected_scalar_int = np.arctanh(x_scalar_int)
    assert isinstance(result_scalar_int, float)
    assert np.isclose(result_scalar_int, expected_scalar_int)

    # 测试 0 维 ti.Field 输入情况
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 0.5
    fill_0d()
    result_0d = bm.atanh(x_0d)
    expected_0d = np.arctanh(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.isclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入情况
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 0.1
            elif i == 1:
                x_1d[i] = 0.3
            else:
                x_1d[i] = 0.5
    fill_1d()
    result_1d = bm.atanh(x_1d)
    expected_1d = np.arctanh(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入情况
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 0.1
            elif i == 0 and j == 1:
                x_2d[i, j] = 0.2
            elif i == 1 and j == 0:
                x_2d[i, j] = 0.3
            else:
                x_2d[i, j] = 0.4
    fill_2d()
    result_2d = bm.atanh(x_2d)
    expected_2d = np.arctanh(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

    # 测试多维 ti.Field 输入情况
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.1
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.2
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.3
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 0.4
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.5
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.6
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.7
            else:
                x_multi[i, j, k] = 0.8
    fill_multi()
    result_multi = bm.atanh(x_multi)
    expected_multi = np.arctanh(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a scalar"):
        bm.atanh(x_invalid_type)

    # 测试 atanh(1.0) 是否返回无穷
    @ti.kernel
    def test_atanh() -> bool:
        y = bm.atanh(1.0)
        return ti.math.isinf(y)

    result_0 = test_atanh()
    assert result_0 == True

    # 测试 atanh(2.0) 是否返回NaN
    @ti.kernel
    def test_atanh_nan() -> bool:
        y = bm.atanh(2.0)
        return ti.math.isnan(y)

    result_nan = test_atanh_nan()
    assert result_nan == True


# 测试 equal 方法
def test_equal():
    # 测试两个形状相同且值相同的 ti.Field
    x_ss = ti.field(dtype=ti.f64, shape=(2, 2))
    y_ss = ti.field(dtype=ti.f64, shape=(2, 2))

    @ti.kernel
    def fill_ss():
        for i, j in x_ss:
            x_ss[i, j] = 1.0
            y_ss[i, j] = 1.0

    fill_ss()
    result_ss = bm.equal(x_ss, y_ss)
    expected_ss = np.ones((2, 2), dtype=bool)
    assert isinstance(result_ss, ti.Field)
    assert result_ss.dtype == ti.u1
    assert np.allclose(result_ss.to_numpy(), expected_ss)

    # 测试两个形状相同但值不同的 ti.Field
    x_sd = ti.field(dtype=ti.f64, shape=(2, 2))
    y_sd = ti.field(dtype=ti.f64, shape=(2, 2))

    @ti.kernel
    def fill_sd():
        for i, j in x_sd:
            x_sd[i, j] = i + j
            y_sd[i, j] = (i + j) * 2

    fill_sd()
    result_sd = bm.equal(x_sd, y_sd)
    expected_sd = np.array([[True, False], [False, False]], dtype=bool)
    assert isinstance(result_sd, ti.Field)
    assert result_sd.dtype == ti.u1
    assert np.allclose(result_sd.to_numpy(), expected_sd)

    # 测试两个形状不同的 ti.Field
    x_ds = ti.field(dtype=ti.f64, shape=(2, 2))
    y_ds = ti.field(dtype=ti.f64, shape=(3, 3))
    with pytest.raises(ValueError, match="Input fields must have the same shape"):
        bm.equal(x_ds, y_ds)

    # 测试输入类型不是 ti.Field
    x_nf = np.array([1, 2, 3])
    y_nf = ti.field(dtype=ti.f64, shape=(3,))
    with pytest.raises(TypeError, match="Both inputs must be ti.Field"):
        bm.equal(x_nf, y_nf)

    # 测试不同数据类型的 ti.Field
    x_dd = ti.field(dtype=ti.f64, shape=(2, 2))
    y_dd = ti.field(dtype=ti.i32, shape=(2, 2))

    @ti.kernel
    def fill_dd():
        for i, j in x_dd:
            x_dd[i, j] = 1.0
            y_dd[i, j] = 1

    fill_dd()
    result_dd = bm.equal(x_dd, y_dd)
    expected_dd = np.ones((2, 2), dtype=bool)
    assert isinstance(result_dd, ti.Field)
    assert result_dd.dtype == ti.u1
    assert np.allclose(result_dd.to_numpy(), expected_dd)


# 测试 exp 方法
def test_exp():
    # 测试标量输入情况 （float类型）
    x_scalar_f = 2.0
    result_scalar = bm.exp(x_scalar_f)
    expected_scalar = np.exp(x_scalar_f)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试标量输入情况 （int 类型）
    x_scalar_int = 2
    result_scalar_int = bm.exp(x_scalar_int)
    expected_scalar_int = np.exp(x_scalar_int)
    assert isinstance(result_scalar_int, float)
    assert np.isclose(result_scalar_int, expected_scalar_int)

    # 测试 0 维 ti.Field 输入情况   
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 3.0
    fill_0d()
    result_0d = bm.exp(x_0d)
    expected_0d = np.exp(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.isclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入情况
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 0.0
            elif i == 1:
                x_1d[i] = 1.0
            else:
                x_1d[i] = 2.0
    fill_1d()
    result_1d = bm.exp(x_1d)
    expected_1d = np.exp(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 0.5
            elif i == 0 and j == 1:
                x_2d[i, j] = 1.5
            elif i == 1 and j == 0:
                x_2d[i, j] = 2.5
            else:
                x_2d[i, j] = 3.5
    fill_2d()

    result_2d = bm.exp(x_2d)
    expected_2d = np.exp(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

    # 测试多维 ti.Field 输入
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.2
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.4
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.6
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 0.8
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 1.0
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 1.2
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 1.4
            else:
                x_multi[i, j, k] = 1.6
    fill_multi()

    result_multi = bm.exp(x_multi)
    expected_multi = np.exp(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.exp(x_invalid_type)

    # 测试输入类型无效的情况
    x_invalid_type = np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a ti.Field or a scalar"):
        bm.exp(x_invalid_type)

    
# 测试 expm1 方法
def test_expm1():
    # 测试大标量输入
    x_scalar = 0.5
    result_scalar = bm.expm1(x_scalar)
    expected_scalar = np.expm1(x_scalar)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试小标量输入（使用泰勒展开）
    x_small_scalar = 1e-6
    result_small_scalar = bm.expm1(x_small_scalar)
    expected_small_scalar = np.expm1(x_small_scalar)
    assert isinstance(result_small_scalar, float)
    assert np.isclose(result_small_scalar, expected_small_scalar, atol=1e-10)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 0.5
    fill_0d()
    result_0d = bm.expm1(x_0d)
    expected_0d = np.expm1(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.isclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 0.1
            elif i == 1:
                x_1d[i] = 0.3
            else:
                x_1d[i] = 0.5
    fill_1d()
    result_1d = bm.expm1(x_1d)
    expected_1d = np.expm1(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 0.1
            elif i == 0 and j == 1:
                x_2d[i, j] = 0.2
            elif i == 1 and j == 0:
                x_2d[i, j] = 0.3
            else:
                x_2d[i, j] = 0.4
    fill_2d()
    result_2d = bm.expm1(x_2d)
    expected_2d = np.expm1(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

    # 测试多维 ti.Field 输入
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.1
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.2
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.3
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 0.4
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.5
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.6
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.7
            else:
                x_multi[i, j, k] = 0.8
    fill_multi()
    result_multi = bm.expm1(x_multi)
    expected_multi = np.expm1(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.expm1(x_invalid_type)

   
# 测试 log 方法
def test_log():
    # 测试标量输入
    x_scalar = 2.0
    result_scalar = bm.log(x_scalar)
    expected_scalar = np.log(x_scalar)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 3.0
    fill_0d()
    result_0d = bm.log(x_0d)
    expected_0d = np.log(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.isclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 1.0
            elif i == 1:
                x_1d[i] = 2.0
            else:
                x_1d[i] = 3.0
    fill_1d()
    result_1d = bm.log(x_1d)
    expected_1d = np.log(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 1.5
            elif i == 0 and j == 1:
                x_2d[i, j] = 2.5
            elif i == 1 and j == 0:
                x_2d[i, j] = 3.5
            else:
                x_2d[i, j] = 4.5
    fill_2d()
    result_2d = bm.log(x_2d)
    expected_2d = np.log(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

    # 测试多维 ti.Field 输入
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 1.2
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 1.4
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 1.6
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 1.8
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 2.0
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 2.2
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 2.4
            else:
                x_multi[i, j, k] = 2.6
    fill_multi()
    result_multi = bm.log(x_multi)
    expected_multi = np.log(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.log(x_invalid_type)

    # 测试log(0)是否返回无穷
    @ti.kernel
    def test_log() -> bool:
        y = bm.log(0)
        return ti.math.isinf(y)

    result_0 = test_log()
    assert result_0 == True

    # 测试log(负数)是否返回NaN
    @ti.kernel
    def test_log_negative() -> bool:
        y = bm.log(-1.0)
        return ti.math.isnan(y)

    result_neg = test_log_negative()
    assert result_neg == True


# 测试 log1p 方法
def test_log1p():
    # 测试大正标量输入
    x_large_pos = 1.0
    res_large_pos = bm.log1p(x_large_pos)
    exp_large_pos = np.log1p(x_large_pos)
    assert isinstance(res_large_pos, float)
    assert np.isclose(res_large_pos, exp_large_pos)

    # 测试小正标量输入
    x_small_pos = 1e-5
    res_small_pos = bm.log1p(x_small_pos)
    exp_small_pos = x_small_pos - (x_small_pos * x_small_pos) / 2 + (x_small_pos * x_small_pos * x_small_pos) / 3
    assert isinstance(res_small_pos, float)
    assert np.isclose(res_small_pos, exp_small_pos)

    # 测试大负标量输入
    x_large_neg = -0.5
    res_large_neg = bm.log1p(x_large_neg)
    exp_large_neg = np.log1p(x_large_neg)
    assert isinstance(res_large_neg, float)
    assert np.isclose(res_large_neg, exp_large_neg)

    # 测试小负标量输入
    x_small_neg = -1e-5
    res_small_neg = bm.log1p(x_small_neg)
    exp_small_neg = x_small_neg - (x_small_neg * x_small_neg) / 2 + (x_small_neg * x_small_neg * x_small_neg) / 3
    assert isinstance(res_small_neg, float)
    assert np.isclose(res_small_neg, exp_small_neg)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 0.5
    fill_0d()
    res_0d = bm.log1p(x_0d)
    exp_0d = np.log1p(x_0d[None])
    assert isinstance(res_0d, ti.Field)
    assert np.isclose(res_0d.to_numpy(), exp_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 1e-5
            elif i == 1:
                x_1d[i] = 1.0
            else:
                x_1d[i] = -1e-5
    fill_1d()
    res_1d = bm.log1p(x_1d)
    exp_1d = np.where(
        np.abs(x_1d.to_numpy()) > 1e-4,
        np.log1p(x_1d.to_numpy()),
        x_1d.to_numpy() - (x_1d.to_numpy()**2)/2 + (x_1d.to_numpy()**3)/3
    )
    assert isinstance(res_1d, ti.Field)
    assert res_1d.dtype == x_1d.dtype
    assert np.allclose(res_1d.to_numpy(), exp_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 1e-6
            elif i == 0 and j == 1:
                x_2d[i, j] = 1.0
            elif i == 1 and j == 0:
                x_2d[i, j] = -1e-6
            else:
                x_2d[i, j] = 2.0
    fill_2d()
    res_2d = bm.log1p(x_2d)
    exp_2d = np.where(
        np.abs(x_2d.to_numpy()) > 1e-4,
        np.log1p(x_2d.to_numpy()),
        x_2d.to_numpy() - (x_2d.to_numpy()**2)/2 + (x_2d.to_numpy()**3)/3
    )
    assert isinstance(res_2d, ti.Field)
    assert res_2d.dtype == x_2d.dtype
    assert np.allclose(res_2d.to_numpy(), exp_2d)

    # 测试输入类型无效的情况
    x_invalid = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.log1p(x_invalid)

    # 测试log1p(-1.0)是否返回无穷
    @ti.kernel
    def test_log1p() -> bool:
        y = bm.log1p(-1.0)
        return ti.math.isinf(y)

    result_0 = test_log1p()         
    assert result_0 == True

    # 测试log1p(小于-1的负数)是否返回NaN
    @ti.kernel
    def test_log1p_negative() -> bool:
        y = bm.log1p(-2.0)
        return ti.math.isnan(y)

    result_neg = test_log1p_negative()
    assert result_neg == True


# 测试 sqrt 方法
def test_sqrt():
    # 测试标量输入
    x_scalar = 2.0
    result_scalar = bm.sqrt(x_scalar)
    expected_scalar = np.sqrt(x_scalar)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 4.0
    fill_0d()
    result_0d = bm.sqrt(x_0d)
    expected_0d = np.sqrt(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.allclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 1.0
            elif i == 1:
                x_1d[i] = 4.0
            else:
                x_1d[i] = 9.0
    fill_1d()
    result_1d = bm.sqrt(x_1d)
    expected_1d = np.sqrt(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 1.0
            elif i == 0 and j == 1:
                x_2d[i, j] = 4.0
            elif i == 1 and j == 0:
                x_2d[i, j] = 9.0
            else:
                x_2d[i, j] = 16.0
    fill_2d()
    result_2d = bm.sqrt(x_2d)
    expected_2d = np.sqrt(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

 # 测试多维 ti.Field 输入情况
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 1.0
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 4.0
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 9.0
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 16.0
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 25.0
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 36.0
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 49.0
            else:
                x_multi[i, j, k] = 64.0
    fill_multi()
    result_multi = bm.sqrt(x_multi)
    expected_multi = np.sqrt(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试单元素 ti.Field 输入
    x_single = ti.field(dtype=ti.f64, shape=(1,))
    x_single.from_numpy(np.array([100.0]))
    result_single = bm.sqrt(x_single)
    expected_single = np.sqrt(100.0)
    assert isinstance(result_single, ti.Field)
    assert np.allclose(result_single.to_numpy(), expected_single)
    
    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.sqrt(x_invalid_type)

    # 测试 qrt (负数)是否返回NaN
    @ti.kernel
    def test_sqrt_negative() -> bool:
        y = bm.sqrt(-1.0)
        return ti.math.isnan(y)

    result_neg = test_sqrt_negative()
    assert result_neg == True


# 测试 sign 方法
def test_sign():
    # 测试正标量输入
    x_scalar_pos = 5.0
    result_scalar_pos = bm.sign(x_scalar_pos)
    expected_scalar_pos = np.sign(x_scalar_pos)
    assert isinstance(result_scalar_pos, float)
    assert np.isclose(result_scalar_pos, expected_scalar_pos)

    # 测试负标量输入
    x_scalar_neg = -3.0
    result_scalar_neg = bm.sign(x_scalar_neg)
    expected_scalar_neg = np.sign(x_scalar_neg)
    assert isinstance(result_scalar_neg, float)
    assert np.isclose(result_scalar_neg, expected_scalar_neg)

    # 测试零标量输入
    x_scalar_zero = 0.0
    result_scalar_zero = bm.sign(x_scalar_zero)
    expected_scalar_zero = np.sign(x_scalar_zero)
    assert isinstance(result_scalar_zero, float)
    assert np.isclose(result_scalar_zero, expected_scalar_zero)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = -2.0
    fill_0d()
    result_0d = bm.sign(x_0d)
    expected_0d = np.sign(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.allclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = 1.0
            elif i == 1:
                x_1d[i] = -1.0
            else:
                x_1d[i] = 0.0
    fill_1d()
    result_1d = bm.sign(x_1d)
    expected_1d = np.sign(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = 3.0
            elif i == 0 and j == 1:
                x_2d[i, j] = -3.0
            elif i == 1 and j == 0:
                x_2d[i, j] = 0.0
            else:
                x_2d[i, j] = 4.0
    fill_2d()
    result_2d = bm.sign(x_2d)
    expected_2d = np.sign(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

 # 测试多维 ti.Field 输入情况
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = 1.0
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = 2.0
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = 3.0
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = 4.0
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 5.0
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 6.0
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 7.0
            else:
                x_multi[i, j, k] = 8.0
    fill_multi()
    result_multi = bm.sign(x_multi)
    expected_multi = np.sign(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.sign(x_invalid_type)


# 测试 tan 方法
def test_tan():
    # 测试标量输入
    x_scalar = np.pi/4
    result_scalar = bm.tan(x_scalar)
    expected_scalar = np.tan(x_scalar)
    assert isinstance(result_scalar,float)
    assert np.isclose(result_scalar,expected_scalar)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = np.pi/3
    fill_0d()
    result_0d = bm.tan(x_0d)
    assert isinstance(result_0d,ti.Field)
    assert np.allclose(result_0d.to_numpy(),np.tan(np.pi/3))

    # 测试 1 维 ti.Field 输入
    x_1d =ti.field(dtype=ti.f64,shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = np.pi/6
            elif i == 1:
                x_1d[i] = np.pi/4
            else:
                x_1d[i] = np.pi/3
    fill_1d()
    result_1d = bm.tan(x_1d)
    expected_1d = np.tan(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)  
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

# 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = np.pi/6
            elif i == 0 and j == 1:
                x_2d[i, j] = np.pi/4
            elif i == 1 and j == 0:
                x_2d[i, j] = np.pi/3
            else:
                x_2d[i, j] = np.pi/2
    fill_2d()
    result_2d = bm.tan(x_2d)
    expected_2d = np.tan(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

# 测试多维 ti.Field 输入
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = np.pi/4
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = np.pi/6
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = np.pi/8
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = np.pi/12
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = np.pi/12
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = np.pi/8
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = np.pi/6
            else:
                x_multi[i, j, k] = np.pi/4
    fill_multi()
    result_multi = bm.tan(x_multi)
    expected_multi = np.tan(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.tan(x_invalid_type)


# 测试 tanh 方法
def test_tanh():
    # 测试标量输入
    x_scalar = 1.0
    result_scalar = bm.tanh(x_scalar)
    expected_scalar = np.tanh(x_scalar)
    assert isinstance(result_scalar, float)
    assert np.isclose(result_scalar, expected_scalar)

    # 测试 0 维 ti.Field 输入
    x_0d = ti.field(dtype=ti.f64, shape=())
    @ti.kernel
    def fill_0d():
        x_0d[None] = 0.5
    fill_0d()
    result_0d = bm.tanh(x_0d)
    expected_0d = np.tanh(x_0d[None])
    assert isinstance(result_0d, ti.Field)
    assert np.isclose(result_0d.to_numpy(), expected_0d)

    # 测试 1 维 ti.Field 输入
    x_1d = ti.field(dtype=ti.f64, shape=(3,))
    @ti.kernel
    def fill_1d():
        for i in x_1d:
            if i == 0:
                x_1d[i] = -0.5
            elif i == 1:
                x_1d[i] = 0.0
            else:
                x_1d[i] = 0.5
    fill_1d()
    result_1d = bm.tanh(x_1d)
    expected_1d = np.tanh(x_1d.to_numpy())
    assert isinstance(result_1d, ti.Field)
    assert result_1d.dtype == x_1d.dtype
    assert np.allclose(result_1d.to_numpy(), expected_1d)

    # 测试 2 维 ti.Field 输入
    x_2d = ti.field(dtype=ti.f64, shape=(2, 2))
    @ti.kernel
    def fill_2d():
        for i, j in x_2d:
            if i == 0 and j == 0:
                x_2d[i, j] = -1.0
            elif i == 0 and j == 1:
                x_2d[i, j] = -0.5
            elif i == 1 and j == 0:
                x_2d[i, j] = 0.5
            else:
                x_2d[i, j] = 1.0
    fill_2d()
    result_2d = bm.tanh(x_2d)
    expected_2d = np.tanh(x_2d.to_numpy())
    assert isinstance(result_2d, ti.Field)
    assert result_2d.dtype == x_2d.dtype
    assert np.allclose(result_2d.to_numpy(), expected_2d)

# 测试多维 ti.Field 输入
    x_multi = ti.field(dtype=ti.f64, shape=(2, 2, 2))
    @ti.kernel
    def fill_multi():
        for i, j, k in ti.ndrange(2, 2, 2):
            if i == 0 and j == 0 and k == 0:
                x_multi[i, j, k] = -0.8
            elif i == 0 and j == 0 and k == 1:
                x_multi[i, j, k] = -0.6
            elif i == 0 and j == 1 and k == 0:
                x_multi[i, j, k] = -0.4
            elif i == 0 and j == 1 and k == 1:
                x_multi[i, j, k] = -0.2
            elif i == 1 and j == 0 and k == 0:
                x_multi[i, j, k] = 0.2
            elif i == 1 and j == 0 and k == 1:
                x_multi[i, j, k] = 0.4
            elif i == 1 and j == 1 and k == 0:
                x_multi[i, j, k] = 0.6
            else:
                x_multi[i, j, k] = 0.8
    fill_multi()
    result_multi = bm.tanh(x_multi)
    expected_multi = np.tanh(x_multi.to_numpy())
    assert isinstance(result_multi, ti.Field)
    assert result_multi.dtype == x_multi.dtype
    assert np.allclose(result_multi.to_numpy(), expected_multi)

    # 测试输入类型无效的情况
    x_invalid_type = "invalid"
    with pytest.raises(TypeError, match="Input must be a ti.Field or a scalar"):
        bm.tanh(x_invalid_type)


# 测试 cross 方法
def test_cross():
    # 测试两个 2D 向量叉积
    vec1_2d = ti.field(dtype=ti.f64, shape=(2,))
    vec2_2d = ti.field(dtype=ti.f64, shape=(2,))

    @ti.kernel
    def fill_2d_vectors():
        vec1_2d[0] = 1.0
        vec1_2d[1] = 2.0
        vec2_2d[0] = 3.0
        vec2_2d[1] = 4.0

    fill_2d_vectors()
    res_2d = bm.cross(vec1_2d, vec2_2d)
    exp_2d = np.array([1.0 * 4.0 - 2.0 * 3.0])
    assert isinstance(res_2d, ti.Field)
    assert res_2d.dtype == vec1_2d.dtype
    assert np.allclose(res_2d.to_numpy(), exp_2d)

    # 测试两个 3D 向量叉积
    vec1_3d = ti.field(dtype=ti.f64, shape=(3,))
    vec2_3d = ti.field(dtype=ti.f64, shape=(3,))

    @ti.kernel
    def fill_3d_vectors():
        vec1_3d[0] = 1.0
        vec1_3d[1] = 0.0
        vec1_3d[2] = 0.0
        vec2_3d[0] = 0.0
        vec2_3d[1] = 1.0
        vec2_3d[2] = 0.0

    fill_3d_vectors()
    res_3d = bm.cross(vec1_3d, vec2_3d)
    exp_3d = np.array([0.0, 0.0, 1.0])
    assert isinstance(res_3d, ti.Field)
    assert res_3d.dtype == vec1_3d.dtype
    assert np.allclose(res_3d.to_numpy(), exp_3d)

    # 测试输入不是 ti.Field 的情况
    vec1_invalid = "invalid"
    vec2_invalid = ti.field(dtype=ti.f64, shape=(2,))
    with pytest.raises(TypeError, match="Both inputs must be ti.Field"):
        bm.cross(vec1_invalid, vec2_invalid)

    # 测试输入形状不匹配的情况
    vec1_mismatch = ti.field(dtype=ti.f64, shape=(2,))
    vec2_mismatch = ti.field(dtype=ti.f64, shape=(3,))
    with pytest.raises(ValueError, match="Input fields must have the same shape"):
        bm.cross(vec1_mismatch, vec2_mismatch)

    # 测试输入不是 2D 或 3D 向量的情况
    vec1_4d = ti.field(dtype=ti.f64, shape=(4,))
    vec2_4d = ti.field(dtype=ti.f64, shape=(4,))
    with pytest.raises(ValueError, match="Input fields must be 1D vectors of length 2 or 3"):
        bm.cross(vec1_4d, vec2_4d)



if __name__ == "__main__":
    pytest.main(["-q", "-s"])
