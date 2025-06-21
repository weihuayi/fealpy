from sympy import symbols
from sympy import Matrix, diff, Function

# 定义符号
x, y, z, l = symbols('x y z l')

# 定义六个函数

L0 = Function('L0')(z) 
L1 = Function('L1')(z)
H0 = Function('H0')(z)
H1 = Function('H1')(z)
H2 = Function('H2')(z)
H3 = Function('H3')(z)


# 计算各函数关于 z 的导数
H0z = diff(H0, z)
H1z = diff(H1, z)
H2z = diff(H2, z)
H3z = diff(H3, z)

# 定义矩阵 Φ
Phi = Matrix([
    [H0, 0, 0, 0, H1, 0, H2, 0, 0, 0, H3, 0],
    [0, H0, 0, -H1, 0, 0, 0, H2, 0, -H3, 0, 0],
    [0, 0, 0, 0, 0, L0, 0, 0, 0, 0, 0, L1],
    [0, -H0z, 0, H1z, 0, 0, 0, -H2z, 0, H3z, 0, 0],
    [H0z, 0, 0, 0, H1z, 0, H2z, 0, 0, 0, H3z, 0],
    [0, 0, 0, 0, 0, L0, 0, 0, 0, 0, 0, L1]
])

#L0 = 1 - z / l
#L1 = z / l
#H0 = 1 - 3 * (z**2 / l**2) + 2 * (z**3 / l**3)
#H1 = z - 2 * (z**2 / l) + (z**3 / l**2)
#H2 = 3 * (z**2 / l**2) - 2 * (z**3 / l**3)
#H3 = -(z**2 / l) + (z**3 / l**2)
