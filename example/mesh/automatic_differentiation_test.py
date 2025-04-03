import torch
import torch.optim as optim
from torch import relu
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')


# 初始化参数
num_points = 20  # 点的数量
R = 5.0  # 边界圆半径
lr = 0.01  # 学习率
num_epochs = 50000  # 迭代次数
spring_k = 0.5  # 弹簧劲度系数
repulsion_strength = 1.0  # 排斥力强度
boundary_penalty_strength = 1.0  # 边界约束强度

# 初始化点的坐标（直接生成叶子张量）
points = bm.random.randn(num_points, 2, requires_grad=True)

# 创建优化器（Adam 优化器）
optimizer = optim.Adam([points], lr=lr)


# 定义能量函数
def compute_energy(points):
    # 1. 粒子间排斥力
    dist_matrix = bm.linalg.norm(points[:, None] - points[None, :], dim=2)
    mask = (dist_matrix > 0)
    repulsion_energy = bm.sum((1 / (dist_matrix[mask] + 1e-6) ** 12)) * repulsion_strength

    # 2. 弹簧势能 (相邻点连接)
    displacements = points[1:] - points[:-1]
    spring_lengths = bm.linalg.norm(displacements, dim=1)
    spring_energy = bm.sum((spring_lengths - 1.0) ** 2) * spring_k

    # 3. 边界约束
    radii = bm.linalg.norm(points, dim=1)
    boundary_penalty = bm.sum(relu(radii - R) ** 2)*boundary_penalty_strength

    total_energy = repulsion_energy + spring_energy + boundary_penalty
    return total_energy


# 显式迭代优化过程
for step in range(num_epochs):
    optimizer.zero_grad()
    energy = compute_energy(points)
    energy.backward()
    optimizer.step()

    # 每隔50步输出
    if (step + 1) % 50 == 0:
        print(f"Step [{step + 1}/{num_epochs}], Energy: {energy.item():.4f}")

# 输出最终结果
print("\n优化后的坐标示例（前5个点）：")
print(bm.to_numpy(points).round(2))

# 绘制结果
plt.figure(figsize=(6, 6))
plt.scatter(bm.to_numpy(points[:, 0]), bm.to_numpy(points[:, 1]), s=100)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Optimized Point Configuration")
plt.show()