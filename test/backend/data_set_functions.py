import numpy as np

# 逐例写出输入与期望输出 (x_np, exp_vals, exp_inds, exp_inv, exp_counts)

# 1) 重复整数
x1 = np.array([1, 2, 6, 4, 2, 3, 2])
exp_vals1 = np.array([1, 2, 3, 4, 6])
exp_inds1 = np.array([0, 1, 5, 3, 2], dtype=np.int64)
exp_inv1 = np.array([0, 1, 4, 3, 1, 2, 1], dtype=np.int64)
exp_counts1 = np.array([1, 3, 1, 1, 1], dtype=np.int64)

# 2) 重复浮点
x2 = np.array([0.1, 0.5, 0.1, 0.8, 0.5, 0.5])
exp_vals2 = np.array([0.1, 0.5, 0.8])
exp_inds2 = np.array([0, 1, 3], dtype=np.int64)
exp_inv2 = np.array([0, 1, 0, 2, 1, 1], dtype=np.int64)
exp_counts2 = np.array([2, 3, 1], dtype=np.int64)

# 3) 布尔值
x3 = np.array([True, False, True, True, False])
exp_vals3 = np.array([False, True])
exp_inds3 = np.array([1, 0], dtype=np.int64)
exp_inv3 = np.array([1, 0, 1, 1, 0], dtype=np.int64)
exp_counts3 = np.array([2, 3], dtype=np.int64)

# 4) 2D 整数数组
x4 = np.array([[1, 2, 3], [2, 3, 4]])
exp_vals4 = np.array([1, 2, 3, 4])
exp_inds4 = np.array([0, 1, 2, 5], dtype=np.int64)
exp_inv4 = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
exp_counts4 = np.array([1, 2, 2, 1], dtype=np.int64)

# 5) 2D 浮点数组
x5 = np.array([[0.1, 0.2], [0.3, 0.1]])
exp_vals5 = np.array([0.1, 0.2, 0.3])
exp_inds5 = np.array([0, 1, 2], dtype=np.int64)
exp_inv5 = np.array([[0, 1], [2, 0]], dtype=np.int64)
exp_counts5 = np.array([2, 1, 1], dtype=np.int64)

# 6) 空数组
x6 = np.array([], dtype=np.int32)
exp_vals6 = np.array([], dtype=np.int32)
exp_inds6 = np.array([], dtype=np.int64)
exp_inv6 = np.array([], dtype=np.int64)
exp_counts6 = np.array([], dtype=np.int64)

# 7) 全相同
x7 = np.array([5, 5, 5, 5, 5])
exp_vals7 = np.array([5])
exp_inds7 = np.array([0], dtype=np.int64)
exp_inv7 = np.array([0, 0, 0, 0, 0], dtype=np.int64)
exp_counts7 = np.array([5], dtype=np.int64)

# 8) 全唯一
x8 = np.array([1, 2, 3, 4, 5])
exp_vals8 = np.array([1, 2, 3, 4, 5])
exp_inds8 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
exp_inv8 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
exp_counts8 = np.array([1, 1, 1, 1, 1], dtype=np.int64)

# 9) ±0 视为相等，返回首现的符号（此处 -0.0）
x9 = np.array([1.0, -0.0, 2.0, +0.0, 1.0])
exp_vals9 = np.array([-0.0, 1.0, 2.0])
exp_inds9 = np.array([1, 0, 2], dtype=np.int64)
exp_inv9 = np.array([1, 0, 2, 0, 1], dtype=np.int64)
exp_counts9 = np.array([2, 2, 1], dtype=np.int64)

# 10) inf 与 -inf
x10 = np.array([np.inf, 1.0, np.inf, -np.inf, 1.0, -np.inf])
exp_vals10 = np.array([-np.inf, 1.0, np.inf])
exp_inds10 = np.array([3, 1, 0], dtype=np.int64)
exp_inv10 = np.array([2, 1, 2, 0, 1, 0], dtype=np.int64)
exp_counts10 = np.array([2, 2, 2], dtype=np.int64)

# 11) 含 NaN（每个 NaN 独立，置后，保持出现顺序）
x11 = np.array([1.0, np.nan, 2.0, np.nan, 1.0])
exp_vals11 = np.array([1.0, 2.0, np.nan, np.nan])
exp_inds11 = np.array([0, 2, 1, 3], dtype=np.int64)
exp_inv11 = np.array([0, 2, 1, 3, 0], dtype=np.int64)
exp_counts11 = np.array([2, 1, 1, 1], dtype=np.int64)

# 12) 复数重复
x12 = np.array([1+2j, 2+3j, 1+2j, 4+5j])
exp_vals12 = np.array([1+2j, 2+3j, 4+5j])
exp_inds12 = np.array([0, 1, 3], dtype=np.int64)
exp_inv12 = np.array([0, 1, 0, 2], dtype=np.int64)
exp_counts12 = np.array([2, 1, 1], dtype=np.int64)

# 13) 复数含 NaN（每个 NaN 独立，置后）
x13 = np.array([1+1j, np.nan+2j, 3+3j, np.nan+2j])
exp_vals13 = np.array([1+1j, 3+3j, np.nan+2j, np.nan+2j])
exp_inds13 = np.array([0, 2, 1, 3], dtype=np.int64)
exp_inv13 = np.array([0, 2, 1, 3], dtype=np.int64)
exp_counts13 = np.array([1, 1, 1, 1], dtype=np.int64)

# 汇总（把四个期望值打包成一个 exp 元组）
unique_all_test_data = [
    (x1,  (exp_vals1,  exp_inds1,  exp_inv1,  exp_counts1)),
    (x2,  (exp_vals2,  exp_inds2,  exp_inv2,  exp_counts2)),
    (x3,  (exp_vals3,  exp_inds3,  exp_inv3,  exp_counts3)),
    (x4,  (exp_vals4,  exp_inds4,  exp_inv4,  exp_counts4)),
    (x5,  (exp_vals5,  exp_inds5,  exp_inv5,  exp_counts5)),
    (x6,  (exp_vals6,  exp_inds6,  exp_inv6,  exp_counts6)),
    (x7,  (exp_vals7,  exp_inds7,  exp_inv7,  exp_counts7)),
    (x8,  (exp_vals8,  exp_inds8,  exp_inv8,  exp_counts8)),
    (x9,  (exp_vals9,  exp_inds9,  exp_inv9,  exp_counts9)),
    (x10, (exp_vals10, exp_inds10, exp_inv10, exp_counts10)),
    (x11, (exp_vals11, exp_inds11, exp_inv11, exp_counts11)),
    (x12, (exp_vals12, exp_inds12, exp_inv12, exp_counts12)),
    (x13, (exp_vals13, exp_inds13, exp_inv13, exp_counts13)),
]