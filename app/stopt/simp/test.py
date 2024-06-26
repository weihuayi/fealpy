import numpy as np

# 材料参数
E = 1.0  # 弹性模量
nu = 0.3  # 泊松比

# 高斯积分点及权重
gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
weights = np.array([1, 1])

# 定义形函数及其导数
def shape_functions(xi, eta):
    N = np.array([
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4
    ])
    dN_dxi = np.array([
        [-(1 - eta) / 4, -(1 - xi) / 4],
        [(1 - eta) / 4, -(1 + xi) / 4],
        [(1 + eta) / 4, (1 + xi) / 4],
        [-(1 + eta) / 4, (1 - xi) / 4]
    ])
    return N, dN_dxi

# 定义弹性矩阵 D
D = E / (1 - nu**2) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# 初始化局部刚度矩阵
KE = np.zeros((8, 8))

# 数值积分计算局部刚度矩阵
for xi in range(len(gauss_points)):
    for eta in range(len(gauss_points)):
        gp_xi = gauss_points[xi]
        gp_eta = gauss_points[eta]
        weight_xi = weights[xi]
        weight_eta = weights[eta]

        # 形函数及其导数
        N, dN_dxi = shape_functions(gp_xi, gp_eta)

        # Jacobi矩阵和Jacobi行列式
        J = dN_dxi.T @ np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ]) / 2
        detJ = np.linalg.det(J)
        J_inv = np.linalg.inv(J)

        # 计算应变位移矩阵 B
        dN_dx = J_inv @ dN_dxi.T
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[0, i]
            B[1, 2*i+1] = dN_dx[1, i]
            B[2, 2*i] = dN_dx[1, i]
            B[2, 2*i+1] = dN_dx[0, i]

        # 计算局部刚度矩阵
        KE += B.T @ D @ B * detJ * weight_xi * weight_eta

print("局部刚度矩阵 KE:")
print(KE.round(4))
