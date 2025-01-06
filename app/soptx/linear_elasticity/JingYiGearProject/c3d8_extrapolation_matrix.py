import sympy as sp
import numpy as np


# 定义符号变量
xi, eta, zeta = sp.symbols('xi eta zeta')  # 实际坐标系
xi_hat, eta_hat, zeta_hat = sp.symbols('xi_hat eta_hat zeta_hat')  # 参考坐标系
sqrt_3 = sp.sqrt(3)

# 定义转换关系
transform = {
    xi_hat: xi * sqrt_3 + (1 - sqrt_3)/2,
    eta_hat: eta * sqrt_3 + (1 - sqrt_3)/2,
    zeta_hat: zeta * sqrt_3 + (1 - sqrt_3)/2
}

# 定义积分点在实际坐标系和参考坐标系下的值
q_points_real = {
    '000': ((3-sqrt_3)/6, (3-sqrt_3)/6, (3-sqrt_3)/6),
    '001': ((3-sqrt_3)/6, (3-sqrt_3)/6, (3+sqrt_3)/6),
    '010': ((3-sqrt_3)/6, (3+sqrt_3)/6, (3-sqrt_3)/6),
    '011': ((3-sqrt_3)/6, (3+sqrt_3)/6, (3+sqrt_3)/6),
    '100': ((3+sqrt_3)/6, (3-sqrt_3)/6, (3-sqrt_3)/6),
    '101': ((3+sqrt_3)/6, (3-sqrt_3)/6, (3+sqrt_3)/6),
    '110': ((3+sqrt_3)/6, (3+sqrt_3)/6, (3-sqrt_3)/6),
    '111': ((3+sqrt_3)/6, (3+sqrt_3)/6, (3+sqrt_3)/6)
}

q_points_ref = {
    '000': (0, 0, 0),
    '001': (0, 0, 1),
    '010': (0, 1, 0),
    '011': (0, 1, 1),
    '100': (1, 0, 0),
    '101': (1, 0, 1),
    '110': (1, 1, 0),
    '111': (1, 1, 1)
}


# 定义形函数
def N1(xi, eta, zeta):
    return (xi_hat * eta_hat * zeta_hat).subs(transform)

def N2(xi, eta, zeta):
    return (xi_hat * eta_hat * (1-zeta_hat)).subs(transform)

def N3(xi, eta, zeta):
    return (xi_hat * (1-eta_hat) * zeta_hat).subs(transform)

def N4(xi, eta, zeta):
    return (xi_hat * (1-eta_hat) * (1-zeta_hat)).subs(transform)

def N5(xi, eta, zeta):
    return ((1-xi_hat) * eta_hat * zeta_hat).subs(transform)

def N6(xi, eta, zeta):
    return ((1-xi_hat) * eta_hat * (1-zeta_hat)).subs(transform)

def N7(xi, eta, zeta):
    return ((1-xi_hat) * (1-eta_hat) * zeta_hat).subs(transform)

def N8(xi, eta, zeta):
    return ((1-xi_hat) * (1-eta_hat) * (1-zeta_hat)).subs(transform)

# 创建一个空的 numpy 数组来存储结果
shape_matrix = np.zeros((8, 8))
point_order = ['000', '001', '010', '011', '100', '101', '110', '111']

# 计算形函数值
print("\n形函数在积分点处的表达式和数值:")
print("-" * 50)
shape_funcs = [N1, N2, N3, N4, N5, N6, N7, N8]

for idx, point_id in enumerate(point_order):
    coords_real = q_points_real[point_id]
    coords_ref = q_points_ref[point_id]
    print(f"\n积分点 {point_id}:")
    print(f"实际坐标: {coords_real}")
    print(f"参考坐标: {coords_ref}")
    
    # 使用实际坐标计算符号表达式
    for i, func in enumerate(shape_funcs, 1):
        # 获取符号表达式
        expr = func(xi, eta, zeta)
        print(f"\nN{i} 符号表达式:")
        print(f"{expr}")
        
        # 计算数值结果 (使用参考坐标系的值)
        xi_val, eta_val, zeta_val = coords_ref
        value = expr.subs({xi: xi_val, eta: eta_val, zeta: zeta_val}).simplify()
        print(f"N{i} 数值结果 = {value}")
        
        # 将结果存入矩阵
        float_val = float(value.evalf())
        shape_matrix[idx, i-1] = float_val

    # 验证形函数和为1
    sum_N = sum(func(coords_real[0], coords_real[1], coords_real[2]) for func in shape_funcs)
    print(f"\n形函数之和 = {sum_N.simplify()} (应该等于1)")
    print("-" * 50)

# 打印数值矩阵
print("\n形函数值矩阵 (8x8):")
print("行: 积分点 [000, 001, 010, 011, 100, 101, 110, 111]")
print("列: 形函数 [N1, N2, N3, N4, N5, N6, N7, N8]")
print("-" * 50)
np.set_printoptions(precision=10, suppress=True, linewidth=200)
print(shape_matrix)

# 验证每行之和是否为1
row_sums = np.sum(shape_matrix, axis=1)
print("\n每行之和:")
print(row_sums)


