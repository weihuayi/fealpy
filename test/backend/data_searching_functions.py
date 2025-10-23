import numpy as np

where_test_data = [
    # 格式：(array, x1, x2, exp)
    (
        np.array([1.0, 2.0, 3.0]) > 1,
        np.array([2.0]),
        np.array([1.0]),
        np.array([1.0, 2.0, 2.0]),
    ),
    (
        np.array([[1, 2], [3, 4]]) < 3,
        np.array([1.0, 2.0]),
        np.array([3, 2]),
        np.array([[1.0, 2.0], [3.0, 2.0]]),
    ),
    (
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) > 5,
        np.array([True, False]),
        np.array([False, True]),
        np.array([[[False, True], [False, True]], [[False, False], [True, False]]]),
    ),
    (
        np.array([0, 3, 0, -3]) > 1,
        1 + 2j,
        -3j,
        np.array([-0.0 - 3.0j, 1.0 + 2.0j, -0.0 - 3.0j, -0.0 - 3.0j]),
    ),
]


nonzero_test_data = [
    # 一维数组：混合零和非零
    (np.array([0, 1, 0, 2, 0]), (np.array([1, 3]),)),
    # 一维数组：全零
    (np.array([0, 0, 0]), (np.array([], dtype=int),)),
    # 一维数组：全非零
    (np.array([5, -1, 3]), (np.array([0, 1, 2]),)),
    # 一维复数数组：包含零和非零复数
    (np.array([0 + 0j, 1 + 2j, 0 + 0j, -3j]), (np.array([1, 3]),)),
    # 一维布尔数组：包含零和非零布尔值
    (np.array([False, True, False, True]), (np.array([1, 3]),)),
    # 二维数组：混合情况
    (
        np.array([[0, 1, 0], [2, 0, -3], [0, 0, 4]]),
        (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 2])),
    ),
    # 二维数组：全零
    (np.zeros((2, 2), dtype=int), (np.array([], dtype=int), np.array([], dtype=int))),
    # 二维数组：全非零
    (np.ones((2, 2), dtype=int), (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))),
    # 二维复数数组
    (np.array([[0 + 0j, 2 + 0j], [0 + 0j, 0 + 0j]]), (np.array([0]), np.array([1]))),
    # 二维布尔数组
    (np.array([[False, True], [True, False]]), (np.array([0, 1]), np.array([1, 0]))),
    # 三维数组：混合情况
    (
        np.array([[[0, 0], [1, 0]], [[0, -2], [0, 3]]]),
        (np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([0, 1, 1])),
    ),
]