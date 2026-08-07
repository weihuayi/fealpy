import torch
import torch.optim as optim

# 初始化可优化变量 a 和 b
a = torch.randn(5, 3, requires_grad=True)  # 假设 a 是3维向量
b = torch.randn(5, 3, requires_grad=True)  # 假设 b 是5维向量

# 定义两个优化器，分别优化 a 和 b
optimizer_a = optim.Adam([a], lr=0.01)  # 只优化 a
optimizer_b = optim.Adam([b], lr=0.01)  # 只优化 b


# 定义能量函数（示例）
def energy1(a, b):
    # 能量函数1：a 需要优化，b 固定
    return torch.sum(a ** 2) + 0.1 * torch.sum(a * b.detach())  # 固定 b，用 detach() 断开梯度


def energy2(a, b):
    # 能量函数2：b 需要优化，a 固定
    return torch.sum(b ** 2) + 0.1 * torch.sum(a.detach() * b)  # 固定 a，用 detach() 断开梯度


# 交替优化循环
for epoch in range(1000):
    # --- 优化 a，固定 b ---
    optimizer_a.zero_grad()  # 清零 a 的梯度（不影响 b）
    loss_a = energy1(a, b)  # 计算能量函数1
    loss_a.backward()  # 反向传播（只计算 a 的梯度）
    optimizer_a.step()  # 更新 a

    # --- 优化 b，固定 a ---
    optimizer_b.zero_grad()  # 清零 b 的梯度（不影响 a）
    loss_b = energy2(a, b)  # 计算能量函数2
    loss_b.backward()  # 反向传播（只计算 b 的梯度）
    optimizer_b.step()  # 更新 b

    # 打印训练进度
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss_a: {loss_a.item():.4f}, Loss_b: {loss_b.item():.4f}")