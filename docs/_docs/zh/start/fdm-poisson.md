---
title: 有限差分求解 Poisson 方程
permalilink: /docs/zh/start/fdm-poisson
key: docs-quick-start-fdm-poisson-zh
author: la
---

# Poisson 方程

$\quad$ 给定区域 $\Omega\subset\mathbb R^d$, 其边界 $\partial \Omega = \Gamma_D \cup \Gamma_N$.
经典的 Poisson 方程形式如下(方便起见, 我们称其为 **A 问题**吧)

$$
\begin{aligned}
    -\Delta u &= f, \quad\text{in }\Omega\\
    u &= g_D, \quad\text{on }\Gamma_D \leftarrow \text{Dirichlet }\\
    \frac{\partial u}{\partial\boldsymbol n} & = g_N, \quad\text{on
    }\Gamma_N\leftarrow \text{Neumann}
\end{aligned}
$$

# 有限差分方法

$\quad$ 给定区域 $\Omega = [a_1, b_1] \times [a_2, b_2]$, 给定 $x$ $y$ 方向的剖分段数分别为
$nx$, $ny$. 则有 

$$
\begin{aligned}
    x_i &= ih_x, i = 0, 1, \cdots, nx, \quad h_x = \frac{b_1 - a_1}{nx}, \\
    y_j &= jh_y, j = 0, 1, \cdots, ny, \quad h_y = \frac{b_2 - a_2}{ny}.
\end{aligned}
$$

使用二阶有限差分算子离散点 $(x_i, y_j)$ 处的拉普拉斯算子, 可得

$$
(\Delta u)_{i,j} = \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2_x} +
\frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2_y}.
$$

当 $h_x = h_y$ 时, 上式可以简写为

$$
-(\Delta u)_{i,j} = \frac{4u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}}{h^2}.
$$

右端 

$$
f_{i,j} = f(x_i, y_j).
$$

可以得到离散的线性系统

$$
\boldsymbol A \boldsymbol u = \boldsymbol f.
$$

其中 $\boldsymbol A \in \mathbb R^{(nx+1)(ny+1), (nx+1)(ny+1)}, \boldsymbol u, \boldsymbol f \in \mathbb R^{(nx+1)(ny+1)}$.

## 边界处理

### Dilichlet 边界处理

当为 $x = 0$ 的边界时, 

$$
u_{0,j} = \frac{u_{-1, j} + u_{1, j}}{2}
$$ 


### Neumann 边界处理

$$
\frac{\partial u}{\partial n}(x_0, y_j) = \frac{u_{0,j}-u_{1,j}}{h} + O(h).
$$

# Fealpy 求解 Poison 问题


给定一个真解为

$$
u  = \cos\pi x\cos\pi y
$$

Poisson  方程, 其定义域为 $[0, 1]^2$, 只有纯的 Dirichlet 边界条件, 下面演示基于
FEALPy 求解这个算例的过程. 

1. 导入创建 pde 模型.

```python
from fealpy.pde.poisson_2d import CosCosData # 导入二维 Poisson 模型实例
pde = CosCosData() # 创建 pde 模型对象
```

2. 生成初始网格.

```python
# 生成网格
from fealpy.functionspace import LagrangeFiniteElementSpace 
mesh = pde.init_mesh(n=4) # 生成初始网格, 其中 4 表示初始网格要加密 4 次
```

3. 创建离散代数系统, 并进行边界条件处理. 

```python
A = mesh.laplace_operator() # 组装系数矩阵对象
b = pde.source(node) # 组装右端向量对象
        
# deal with boundary condition
isBDNode = mesh.ds.boundary_node_flag()
idx, = np.nonzero(isBDNode)

x = np.zeros(NN, dtype=mesh.ftype)
x[idx] = pde.dirichlet(node[idx])

b = b - A@x
b[idx] = pde.dirichlet(node[idx])

bdIdx = np.zeros((A.shape[0],), dtype=mesh.itype)
bdIdx[idx] = 1

Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[1])
AD = T@A@T + Tbd
```

4. 求解离散系统.

```python
# 导入稀疏线性代数系统求解函数
from scipy.sparse.linalg import spsolve
uh[:] = spsolve(AD, b).reshape(-1)
```

5. 计算 L2 误差.

```python
L2Error = np.sqrt(np.sum(hx*hy*(self.uI - uh)**2))
```

6. 画出解函数和网格图像

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
axes = fig.add_subplot(1, 2, 1, projection='3d')
uh.add_plot(axes, cmap='rainbow')
axes = fig.add_subplot(1, 2, 2)
mesh.add_plot(axes)
```

