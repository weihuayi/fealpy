#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sun 08 Sep 2024 04:33:43 PM CST
	@bref 
	@ref 
'''  
import sympy as sp
import numpy as np

# 第一步：定义符号变量
x1, x2 = sp.symbols('x1 x2')

# 第二步：定义矩阵A
A = sp.Matrix([[1 + x1**2, 0], [0, 1 + x2**2]])

# 第三步：使用lambdify将A转换为numpy函数，支持输入数组
A_func = sp.lambdify((x1, x2), A, modules='numpy')

# 假设输入点是一组 (NC, NQ, 2) 的 numpy 数组
points = np.random.randn(3, 4, 2)  # 示例数据，假设 (NC, NQ, 2)

# 第四步：分离x1和x2的值
x1_vals = points[..., 0]
x2_vals = points[..., 1]

# 第五步：使用lambdify得到的函数，直接进行广播计算
A_result = np.array(A_func(x1_vals, x2_vals))

# 结果 now contains the (NC, NQ, 2, 2) shaped matrix values.
print(A_result)
