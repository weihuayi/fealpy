import numpy as np

all_test_data = [
    # 格式：(input, axis, keepdims, exp)
    # 一维数组
    (np.array([1, 1, 1]), None, False, np.array(True)),       # 全 True
    (np.array([1, 0, 1]), None, False, np.array(False)),      # 含 False
    # 空数组
    (np.array([]), None, False, np.array(True)),              # 空数组 → True
    # 二维数组，axis 测试
    (np.array([[1, 1], [1, 0]]), None, False, np.array(False)),             # 整体归约
    (np.array([[1, 1], [1, 0]]), 0, False, np.array([True, False])),        # 按列
    (np.array([[1, 1], [1, 0]]), 1, False, np.array([True, False])),        # 按行
    # keepdims=True
    (np.array([[1, 1], [1, 0]]), 0, True, np.array([[True, False]])),
    (np.array([[1, 1], [1, 0]]), 1, True, np.array([[True], [False]])),
    # 特殊值：NaN / Inf
    (np.array([np.nan, np.inf, -np.inf]), None, False, np.array(True)),
    # 复数：非零为 True，零为 False
    (np.array([1+0j, 2j]), None, False, np.array(True)),
    (np.array([1+0j, 0+0j]), None, False, np.array(False)),
    # 高维数组
    (np.array([[[1, 1], [1, 1]], [[1, 0], [1, 1]]]), (0, 1), False, np.array([True, False])),
    # keepdims + 高维
    (np.array([[[1, 1], [1, 1]], [[1, 0], [1, 1]]]), (0, 1), True, np.array([[[True, False]]])),
]


any_test_data = [
    # 一维数组
    (np.array([0, 0, 0]), None, False, np.array(False)),     # 全 False
    (np.array([0, 1, 0]), None, False, np.array(True)),      # 含 True
    # 空数组 → False
    (np.array([]), None, False, np.array(False)),
    # 二维数组，axis 测试
    (np.array([[0, 0], [0, 1]]), None, False, np.array(True)),             # 整体归约
    (np.array([[0, 0], [0, 1]]), 0, False, np.array([False, True])),       # 按列
    (np.array([[0, 0], [0, 1]]), 1, False, np.array([False, True])),       # 按行
    # keepdims=True
    (np.array([[0, 0], [0, 1]]), 0, True, np.array([[False, True]])),
    (np.array([[0, 0], [0, 1]]), 1, True, np.array([[False], [True]])),
    # 特殊值：NaN / Inf
    (np.array([np.nan, np.inf, -np.inf]), None, False, np.array(True)),
    # 复数：非零为 True，零为 False
    (np.array([0+0j, 0+0j]), None, False, np.array(False)),
    (np.array([0+0j, 1+0j]), None, False, np.array(True)),
    # 高维数组
    (np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]), (0, 1), False, np.array([False, True])),
    # keepdims + 高维
    (np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]]), (0, 1), True, np.array([[[False, True]]])),
]


diff_test_data = [
    # 格式：(input, axis, n, prepend, append, exp)
    # ===== 一维数组 =====
    # 基础一阶差分
    (np.array([1, 2, 3]), -1, 1, None, None, np.array([1, 1])),
    # 二阶差分
    (np.array([1, 2, 4, 7]), -1, 2, None, None, np.array([1, 1])),
    # 带 prepend
    (np.array([1, 2, 4]), -1, 1, np.array([0]), None, np.array([1, 1, 2])),
    # 带 append
    (np.array([1, 2, 4]), -1, 1, None, np.array([10]), np.array([1, 2, 6])),
    # 同时带 prepend 和 append
    (np.array([1, 2, 4]), -1, 1, np.array([0]), np.array([10]), np.array([1, 1, 2, 6])),

    # ===== 二维数组 =====
    # axis=1 按列差分
    (np.array([[1, 3, 6, 10],
               [0, 5, 6, 8]]), 1, 1, None, None,
     np.array([[2, 3, 4],
               [5, 1, 2]])),
    # axis=0 按行差分
    (np.array([[1, 2],
               [3, 5],
               [6, 9]]), 0, 1, None, None,
     np.array([[2, 3],
               [3, 4]])),
    # axis=0 带 prepend
    (np.array([[1, 2],
               [3, 5]]), 0, 1, np.array([[0, 0]]), None,
     np.array([[1, 2],
               [2, 3]])),

    # ===== 布尔数组 =====
    (np.array([True, True, False]), -1, 1, None, None, np.array([0, -1])),

    # ===== 复数数组 =====
    (np.array([1+1j, 2+3j, 5+8j]), -1, 1, None, None, np.array([1+2j, 3+5j])),

    # ===== 边界情况 =====
    (np.array([42]), -1, 1, None, None, np.array([])),   # 单元素
    (np.array([]), -1, 1, None, None, np.array([])),    # 空数组
]
