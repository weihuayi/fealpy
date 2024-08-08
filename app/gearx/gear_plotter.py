import numpy as np
import matplotlib.pyplot as plt

# 定义渐开线的函数，增加旋转方向的控制参数 direction
def involute_curve(phi, r_b, theta_offset, direction=1):
    x = r_b * (np.cos(direction * phi + theta_offset) + direction * phi * np.sin(direction * phi + theta_offset))
    y = r_b * (np.sin(direction * phi + theta_offset) - direction * phi * np.cos(direction * phi + theta_offset))
    return x, y

# 参数
m = 5
z = 22
alpha = 20
r_p = m*z
r_b = r_p*np.cos(np.radians(alpha))
r_a = r_p + m
r_f = r_p - 1.25*m

phi_range = np.linspace(0, np.pi/4, 100)  # 参数范围
theta_offsets = np.linspace(0, 2*np.pi, z)  # 四条渐开线的起始角度
delta_theta = np.pi/z/1.5

# 选择旋转方向: 1 为逆时针，-1 为顺时针
direction = -1  # 修改为 -1 可以得到顺时针旋转的渐开线

# 绘图
plt.figure(figsize=(8, 8))
for theta_offset in theta_offsets:
    x, y = involute_curve(phi_range, r_b, theta_offset+delta_theta, direction)
    plt.plot(x, y)


direction = 1  # 修改为 -1 可以得到顺时针旋转的渐开线

for theta_offset in theta_offsets:
    x, y = involute_curve(phi_range, r_b, theta_offset-delta_theta, direction)
    plt.plot(x, y)

# 绘制基圆
theta = np.linspace(0, 2*np.pi, 300)
x_circle = r_b * np.cos(theta)
y_circle = r_b * np.sin(theta)
plt.plot(x_circle, y_circle)

# 绘制分度圆
r_p = r_b/np.cos(np.radians(alpha))
theta = np.linspace(0, 2*np.pi, 300)
x_circle = r_p * np.cos(theta)
y_circle = r_p * np.sin(theta)
plt.plot(x_circle, y_circle)

# 绘制齿顶圆
theta = np.linspace(0, 2*np.pi, 300)
x_circle = r_a * np.cos(theta)
y_circle = r_a * np.sin(theta)
plt.plot(x_circle, y_circle)

# 绘制齿根圆
theta = np.linspace(0, 2*np.pi, 300)
x_circle = r_f * np.cos(theta)
y_circle = r_f * np.sin(theta)
plt.plot(x_circle, y_circle)

# 图形设置
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
