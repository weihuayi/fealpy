import taichi as ti
import math

"""
例子来源于 https://github.com/taichi-dev/taichi
"""

ti.init(arch=ti.cuda)

n = 512
steps = 32
eps = 1e-5

# 全局变量
b = ti.field(float, (n, n))
x = ti.field(float, (n, n)) # 存储未知量
d = ti.field(float, (n, n)) # 残量
r = ti.field(float, (n, n)) # 残量

# 元编程
@ti.func
def c(x: ti.template(), i, j):
    return x[i, j] if 0 <= i < n and 0 <= j < n else 0.0

# 实现一个无矩阵的矩阵乘向量
@ti.func
def A(x: ti.template(), I):
    i, j = I
    return x[i, j] * 4 - c(x, i - 1, j) - c(x, i + 1, j) \
            - c(x, i, j - 1) - c(x, i, j + 1)

# 计算初始的残量
@ti.kernel
def init():
    for I in ti.grouped(x):
        d[I] = b[I] - A(x, I) # 计算 b - Ax
        r[I] = d[I]

# CG 的迭代步
@ti.kernel
def substep():
    alpha, beta, dAd = 0.0, 0.0, eps
    for I in ti.grouped(x):
        dAd += d[I] * A(d, I)
    for I in ti.grouped(x):
        alpha += r[I]**2 / dAd
    for I in ti.grouped(x):
        x[I] = x[I] + alpha * d[I]
        r[I] = r[I] - alpha * A(d, I)
        beta += r[I]**2 / ((alpha + eps) * dAd)
    for I in ti.grouped(x):
        d[I] = r[I] + beta * d[I]


gui = ti.GUI('Possion Solver', (n, n))
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB:
            b[int(e.pos[0] * n), int(e.pos[1] * n)] += 0.75
            init()
    for i in range(steps):
        substep()
    gui.set_image(x)
    gui.show()

