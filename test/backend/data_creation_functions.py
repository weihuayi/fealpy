import numpy as np

arange_test_data = [
    # 格式：(start, stop, step, exp)
    # ===== 基础功能 =====
    (5, None, None, np.array([0, 1, 2, 3, 4])),  # 单参数整型
    (2.0, 10.0, None, np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),  # 双参数浮点型
    (0.5, 10, 3, np.array([0.5, 3.5, 6.5, 9.5])),  # 三参数浮点型
    # ===== 边界情况 =====
    (0, 4, 6, np.array([0])),  # 步长大于区间
    (4, 2, 1, np.array([])),  # start > stop
    (2, 2, 1, np.array([])),  # start = stop
    (5, 0, -1, np.array([5, 4, 3, 2, 1])),  # 负步长正常
    (0, 5, -1, np.array([])),  # 负步长但方向不对
    (0, 1000000, 100000, np.array([0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000])),  # 大数测试
    # ===== 负数区间 =====
    (-5, 0.0, 2, np.array([-5.0, -3.0, -1.0])),  # 负起点
    (2, -5, -1, np.array([2, 1, 0, -1, -2, -3, -4])),  # 负起点+负方向
    # ===== 精度问题 =====
    (0.0, 0.3, 0.05, np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])),  # 很小的浮点步长
    (1e-10, 1e-9, 1e-10, np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10])),  # 极小数值
    # ===== 负浮点步长 =====
    (1.0, -1.0, -0.5, np.array([1.0, 0.5, 0.0, -0.5])),
    # ====== 单元素 =====
    (3, 4, 10, np.array([3])),  # 只生成一个元素
    (0.5, 0.6, 0.2, np.array([0.5])),  # 浮点单元素
]


asarray_test_data = [
    # 格式：(x, dtype, exp)

    # ===== 数组 =====
    (np.array([1, 2, 3]), None, np.array([1, 2, 3], dtype=np.int64)),  # 整型数组
    (np.array([1.1, 2.2, 3.3]), None, np.array([1.1, 2.2, 3.3], dtype=np.float64)),  # 浮点型数组
    (np.array([True, False, True]), None, np.array([True, False, True], dtype=np.bool_)),  # 布尔数组
    (np.array([1+2j, 3+4j]), None, np.array([1+2j, 3+4j], dtype=np.complex128)),  # 复数数组

    # ===== Python 标量 =====
    (1, None, np.array(1, dtype=np.int64)),  # 整型标量
    (3.14, None, np.array(3.14, dtype=np.float32)),  # 浮点标量
    (True, None, np.array(True, dtype=np.bool_)),  # 布尔标量
    (2+3j, None, np.array(2+3j, dtype=np.complex64)),  # 复数标量

    # ===== Python 序列 =====
    ([1, 2, 3], None, np.array([1, 2, 3], dtype=np.int64)),  # 整型列表
    ([1.1, 2.2, 3.3], None, np.array([1.1, 2.2, 3.3], dtype=np.float32)),  # 浮点列表
    ([True, False], None, np.array([True, False], dtype=np.bool_)),  # 布尔列表
    ([1+1j, 2+2j], None, np.array([1+1j, 2+2j], dtype=np.complex64)),  # 复数列表

    # ===== 嵌套序列 =====
    ([[1, 2], [3, 4]], None, np.array([[1, 2], [3, 4]], dtype=np.int64)),  # 二维整型
    ([[1.1, 2.2], [3.3, 4.4]], None, np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)),  # 二维浮点
    ([[True, False], [False, True]], None, np.array([[True, False], [False, True]], dtype=np.bool_)),  # 二维布尔
    ([[1+1j, 2+2j], [3+3j, 4+4j]], None, np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)),  # 二维复数

    # ===== 混合类型推断 =====
    ([1, True], None, np.array([1, 1], dtype=np.int64)),  # int + bool → int
    ([1, 2.5], None, np.array([1.0, 2.5], dtype=np.float32)),  # int + float → float
    ([1, 2+3j], None, np.array([1+0j, 2+3j], dtype=np.complex64)),  # int + complex → complex
    ([True, 2.5], None, np.array([1.0, 2.5], dtype=np.float32)),  # bool + float → float
    ([True, 2+3j], None, np.array([1+0j, 2+3j], dtype=np.complex64)),  # bool + complex → complex
    ([2.5, 1+3j], None, np.array([2.5+0j, 1+3j], dtype=np.complex64)),  # float + complex → complex

    # ===== dtype 指定 =====
    ([1, 2, 3], 'float32', np.array([1, 2, 3], dtype=np.float32)),  # 强制 float32
    ([1.1, 2.2], 'int32', np.array([1, 2], dtype=np.int32)),  # 强制 int32（截断）
    ([True, False], 'int8', np.array([1, 0], dtype=np.int8)),  # bool → int8
    ([1+2j, 3+4j], 'complex64', np.array([1+2j, 3+4j], dtype=np.complex64)),  # complex64
]


eye_test_data = [
    # 格式：(n_rows, n_cols, k, exp)
    # 基础：方阵，主对角线
    (3, None, 0, np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])),
    # 非方阵：行数 != 列数
    (2, 4, 0, np.array([[1,0,0,0],
                        [0,1,0,0]])),
    # 上对角线 k=1
    (3, 3, 1, np.array([[0,1,0],
                        [0,0,1],
                        [0,0,0]])),
    # 下对角线 k=-1
    (3, None, -1, np.array([[0,0,0],
                            [1,0,0],
                            [0,1,0]])),
    # k 超过范围（正数）
    (3, 3, 5, np.zeros((3,3))),
    # k 超过范围（负数）
    (3, 3, -5, np.zeros((3,3))),
    # 复数 dtype
    (2, 2, 0, np.array([[1+0j, 0+0j],
                        [0+0j, 1+0j]])),
]
