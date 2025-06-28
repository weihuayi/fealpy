import numpy as np


arr=np.arange(-10, 10, 1)

print(arr)

t=np.tril(3, k=-1)
a = np.array([[1, 2, 3, 6],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# 返回下三角矩阵（包括对角线)
lower_triangle = np.tril(a)
print(lower_triangle)

a = np.array([[-1, -2, -3, -5],
             [-1, -2, -3, 4],
              [-2, -3, 4, 5]])

# 返回绝对值
print(np.abs(a))