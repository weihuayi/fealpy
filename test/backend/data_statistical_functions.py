import numpy as np

max_test_data = [
    # 格式：(input, axis, keepdims, expected)
    # ===== 基础功能 =====
    (np.array([1, 2, 3]), None, False, 3),  # 无轴指定
    (np.array([[1, 2], [3, 4]]), None, False, 4),  # 无轴指定
    (np.array([1, 2, 3]), 0, False, 3),  # 轴0
    (np.array([[1, 2], [3, 4]]), 0, False, np.array([3, 4])),  # 轴0
    (np.array([[1, 2], [3, 4]]), 1, False, np.array([2, 4])),  # 轴1
    #TODO (np.array([[1, 2], [3, 4]]), (0,1), False, 4),  # 多轴指定 # PyTorch不支持传入元组（tuple）类型的axis参数。
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, True, np.array([3])),  # 无轴指定+keepdims
    (np.array([[1, 2], [3, 4]]), None, True, np.array([[4]])),  # 无轴指定+keepdims
    (np.array([1, 2, 3]), 0, True, np.array([3])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 0, True, np.array([[3, 4]])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 1, True, np.array([[2], [4]])),  # 轴1+keepdims
    # # # ===== 特殊情况 =====
    (np.array([np.nan, 1, 2]), None, False, np.nan),  # 包含NaN
    (np.array([np.nan, np.nan, np.nan]), None, False, np.nan),  # 全NaN
]  
 
min_test_data = [
    # 格式：(input, axis, keepdims, expected)
    # ===== 基础功能 =====
    (np.array([1, 2, 3]), None, False, 1),  # 无轴指定
    (np.array([[1, 2], [3, 4]]), None, False, 1),  # 无轴指定
    (np.array([1, 2, 3]), 0, False, 1),  # 轴0
    (np.array([[1, 2], [3, 4]]), 0, False, np.array([1, 2])),  # 轴0
    (np.array([[1, 2], [3, 4]]), 1, False, np.array([1, 3])),  # 轴1
    # (np.array([[1, 2], [3, 4]]), (0,1), False, 1),  # 多轴指定 # PyTorch不支持传入元组（tuple）类型的axis参数。
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, True, np.array([1])),  # 无轴指定+keepdims
    (np.array([[1, 2], [3, 4]]), None, True, np.array([[1]])),  # 无轴指定+keepdims
    (np.array([1, 2, 3]), 0, True, np.array([1])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 0, True, np.array([[1, 2]])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 1, True, np.array([[1], [3]])),  # 轴1+keepdims
    # # # ===== 特殊情况 =====
    (np.array([np.nan, 1, 2]), None, False, np.nan),  # 包含NaN
    (np.array([np.nan, np.nan, np.nan]), None, False, np.nan),  # 全NaN
]

mean_test_data = [
    # 格式：(input, axis, keepdims, expected)
    # ===== 基础功能 =====
    (np.array([1, 2, 3]), None, False, 2),  # 无轴指定
    (np.array([[1, 2], [3, 4]]), None, False, 2.5),  # 无轴指定
    (np.array([1, 2, 3]), 0, False, 2),  # 轴0
    (np.array([[1, 2], [3, 4]]), 0, False, np.array([2, 3])),  # 轴0
    (np.array([[1, 2], [3, 4]]), 1, False, np.array([1.5, 3.5])),  # 轴1
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, True, np.array([2])),  # 无轴指定+keepdims
    (np.array([[1, 2], [3, 4]]), None, True, np.array([[2.5]])),  # 无轴指定+keepdims
    (np.array([1, 2, 3]), 0, True, np.array([2])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 0, True, np.array([[2, 3]])),  # 轴0+keepdims
    (np.array([[1, 2], [3, 4]]), 1, True, np.array([[1.5], [3.5]])),  # 轴1+keepdims
    # # # ===== 特殊情况 =====
    (np.array([np.nan, 1, 2]), None, False, np.nan),  # 包含NaN
    (np.array([np.nan, np.nan, np.nan]), None, False, np.nan),  # 全NaN
]

prod_test_data = [
    # 格式：(input, axis, keepdims, expected)
    # ===== 基础功能 =====
    (np.array([1.2, 2, 3]), None, False, 7.2),  # 1维数组无轴指定
    #TODO (np.array([[1, 2], [3, 4]]), (0, 1), False, 24),  # 2维数组全维度（仅NumPy支持，PyTorch需特殊处理）
    (np.array([1, 2, 3]), 0, False, 6),  # 轴0（1维）
    (np.array([[1, 2], [3, 4]]), 0, False, np.array([3, 8])),  # 轴0（2维）
    (np.array([[1, 2], [3, 4]]), 1, False, np.array([2, 12])),  # 轴1（2维）
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, True, np.array([6])),  # 1维数组无轴+keepdims
    #TODO (np.array([[1, 2], [3, 4]]), (0, 1), True, np.array([[24]])),  # 2维全维度+keepdims
    (np.array([1, 2, 3]), 0, True, np.array([6])),  # 轴0+keepdims（1维）
    (np.array([[1, 2], [3, 4]]), 0, True, np.array([[3, 8]])),  # 轴0+keepdims（2维）
    (np.array([[1, 2], [3, 4]]), 1, True, np.array([[2], [12]])),  # 轴1+keepdims（2维）
    # ===== 特殊情况（包含NaN）=====
    (np.array([np.nan, 1, 2]), None, False, np.nan),
    (np.array([np.nan, np.nan, np.nan]), None, False, np.nan),
    (np.array([]),None,False,1),
]

sum_test_data = [
    # 格式：(input, axis, keepdims, expected)
    # ===== 基础功能 =====
    (np.array([1, 2, 3]), None, False, 6),  # 1维数组无轴指定
    (np.array([[1, 2], [3, 4]]), (0, 1), False, 10),  # 2维数组全维度（仅NumPy支持，PyTorch需特殊处理）
    (np.array([1, 2, 3]), 0, False, 6),  # 轴0（1维）
    (np.array([[1, 2], [3, 4]]), 0, False, np.array([4, 6])),  # 轴0（2维）
    (np.array([[1, 2], [3, 4]]), 1, False, np.array([3, 7])),  # 轴1（2维）
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, True, np.array([6])),  # 1维数组无轴+keepdims
    (np.array([[1, 2], [3, 4]]), (0, 1), True, np.array([[10]])),  # 2维全维度+keepdims
    (np.array([1, 2, 3]), 0, True, np.array([6])),  # 轴0+keepdims（1维）
    (np.array([[1, 2], [3, 4]]), 0, True, np.array([[4, 6]])),  # 轴0+keepdims（2维）
    (np.array([[1, 2], [3, 4]]), 1, True, np.array([[3], [7]])),  # 轴1+keepdims（2维）
    # ===== 特殊情况（包含NaN）=====
    (np.array([np.nan, 1, 2]), None, False, np.nan),
    (np.array([np.nan, np.nan, np.nan]), None, False, np.nan),
    (np.array([]),None,False,0), # 空数组情况
]
    
std_test_data = [
# 格式：(input, axis, correction, keepdims, expected)
# ===== 基础功能 =====
    (np.array([1, 2, 3]), None, 0.0, False, np.sqrt(2/3)),  # 无轴指定,c = 0.0
    (np.array([1, 2, 3]), None, 1.0, False, np.sqrt(2/2)),  # 无轴指定，c = 1.0
    (np.array([[1, 2], [3, 4]]), None, 0.0, False, np.sqrt(5/4)),  # 无轴指定，c = 0.0
    (np.array([[1, 2],[3, 4]]),None, 1.0, False,np.sqrt(5/3)), # 无轴指定,c = 1.0
    (np.array([1, 2, 3]), 0, 0.0, False, np.sqrt(2/3)),  # 轴0,c = 0.0
    (np.array([1, 2, 3]), 0, 1.0, False, np.sqrt(2/2)),  # 轴0,c = 1.0
    (np.array([[1, 2], [3, 4]]), 0, 0.0, False, np.array([np.sqrt(1), np.sqrt(1)])),  # 轴0,c = 0.0
    (np.array([[1, 2], [3, 4]]), 0, 1.0, False, np.array([np.sqrt(2), np.sqrt(2)])),  # 轴1,c = 0.0
    (np.array([[1, 2], [3, 4]]), 1, 0.0, False, np.array([np.sqrt(1/4), np.sqrt(1/4)])),  # 轴1,c = 1.0
    (np.array([[1, 2], [3, 4]]), 1, 1.0, False, np.array([np.sqrt(1/2), np.sqrt(1/2)])),  # 轴1,c = 0.0
    # ===== keepdims为True的情况 =====
    (np.array([1, 2, 3]), None, 0.0, True, np.array([np.sqrt(2/3)])),  # 无轴指定+keepdims,c = 0.0
    (np.array([1, 2, 3]), None, 1.0, True, np.array([np.sqrt(2/2)])),  # 无轴指定+keepdims,c = 1.0
    (np.array([[1, 2], [3, 4]]), None, 0.0, True, np.array([[np.sqrt(5/4)]])),  # 无轴指定+keepdims,c = 0.0
    (np.array([[1, 2], [3, 4]]), None, 1.0, True, np.array([[np.sqrt(5/3)]])),   # 无轴指定+keepdims,c = 1.0
    (np.array([1, 2, 3]), 0, 0.0, True, np.array([np.sqrt(2/3)])),  # 轴0+keepdims,c = 0.0
    (np.array([1, 2, 3]), 0, 1.0, True, np.array([np.sqrt(2/2)])),  # 轴0+keepdims,c = 1.0
    (np.array([[1, 2], [3, 4]]), 0, 0.0, True, np.array([[np.sqrt(1), np.sqrt(1)]])),  # 轴0+keepdims,c = 0.0
    (np.array([[1, 2], [3, 4]]), 0, 1.0, True, np.array([[np.sqrt(2), np.sqrt(2)]])),  # 轴0+keepdims,c = 1.0
    (np.array([[1, 2], [3, 4]]), 1, 0.0, True, np.array([[np.sqrt(1/4)], [np.sqrt(1/4)]])),  # 轴1+keepdims,c = 0.0
    (np.array([[1, 2], [3, 4]]), 1, 1.0, True, np.array([[np.sqrt(1/2)], [np.sqrt(1/2)]])),  # 轴1+keepdims,c = 1.0
    # ===== 特殊情况 =====
    (np.array([np.nan, 1, 2]), None, 0.0, False, np.nan),  # 包含NaN
    (np.array([np.nan, np.nan, np.nan]), None, 0.0, False, np.nan),  # 全NaN
    (np.array([]),None, 0.0, False,np.nan), # 空数组情况(N=0的情况)
]
    
    
    
