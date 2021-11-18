---
title: 非线性 Poisson 方程数值求解
permalink: /docs/zh/example/num-non-linear-poisson
key: docs-num-non-linear-poisson-zh
---

非线性方程在实际问题中经常出现，这里详细介绍求解非线性 Poisson
方程的典型方法，如

* Newton-Galerkin 方法
    - 先应用 Newton 方法线性化连续的弱形式，再用有限元离散次迭代求解
    - 先应用有限元离散连续的弱形式，再应用 Newton 方法迭代求解
* Picard 迭代方法
    - 又称定点迭代

## Newton-Galerkin 方法

首先给出一个扩散系数为非线性的例子. 求解区域记为 $\Omega$, 边界
$\partial\Omega = \Gamma_D\cup\Gamma_N\cup\Gamma_R$.

$$
-\nabla\left(a(u)\nabla u\right) = f
$$

满足如下的边界条件：

$$
\begin{aligned}
u =& g_D, \quad\text{on }\Gamma_D\leftarrow \text{Dirichlet } \\
a(u)\frac{\partial u}{\partial\boldsymbol n}  =& g_N, 
\quad\text{on }\Gamma_N \leftarrow \text{Neumann} \\
a(u)\frac{\partial u}{\partial\boldsymbol n} + \kappa u =& g_R, 
\quad\text{on }\Gamma_R \leftarrow \text{ Robin}
\end{aligned}
$$

在 Poisson 方程两端分别乘以测试函数 $v \in H_{D,0}^1(\Omega)$, 利用分部积分，
可得到其对应的**连续弱形式**

$$
(a(u)\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_R} = 
(f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

设 $u^0$ 是 $u$ 的一个逼近，记 $\delta u = u - u^0$, 代入连续弱形式

$$
(a(u^0+\delta u)\nabla (u^0+\delta u),\nabla v)+
<\kappa (u^0+\delta u), v>_{\Gamma_R} = 
(f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

其中 $a(u^0+\delta u)$ 在 $u^0$ 处 Taylor 展开，可得

$$
a(u^0 + \delta u) = a(u^0) + a_u'(u^0)\delta u + \mathcal O(\delta u^2)
$$

替换连续弱形式中的 $a(u^0+\delta u)$, 并忽略掉其中 $\mathcal O(\delta u^2)$ 
可得

$$
\begin{aligned}
& (a(u^0)\nabla\delta u, \nabla v) + (a_u'(u^0)\nabla u^0\cdot\delta u, \nabla v) 
+ <\kappa\delta u, v>_{\Gamma_R} \\
=&  (f,v) - (a(u^0)\nabla u^0, \nabla v) - 
<\kappa u^0, v>_{\Gamma_R} + <g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
\end{aligned}
$$

给定求解区域 $\Omega$ 上的网格离散 $\mathcal T = \{\tau\}$, 
构造 $N$ 维的有限维空间 $V_h$，
其 $N$ 个**全局基函数**组成的**行向量函数**记为

$$
\boldsymbol\phi(\boldsymbol x) = 
\left[
\phi_0(\boldsymbol x), \phi_1(\boldsymbol x), \cdots, \phi_{N-1}(\boldsymbol x)
\right], \boldsymbol x \in \Omega 
$$

对于有限元程序设计实现来说，并不会显式构造出**全局基函数**，实际基函数的求值计
算都发生网格单元或网格单元的边界上。设每个网格单元 $\tau$ 上**局部基函数**个数为 
$l$ 个，其组成的**行向量函数**记为

$$
\boldsymbol\varphi(\boldsymbol x) = 
\left[
\varphi_0(\boldsymbol x), \varphi_1(\boldsymbol x), 
\cdots, \varphi_{l-1}(\boldsymbol x)
\right], \boldsymbol x \in \tau
$$

所有基函数梯度对应的向量

$$
\nabla \boldsymbol\varphi(\boldsymbol x) = 
\left[
\nabla \varphi_0(\boldsymbol x), \nabla \varphi_1(\boldsymbol x), 
\cdots, \nabla \varphi_{l-1}(\boldsymbol x)
\right], \boldsymbol x \in \tau
$$

其中 

$$
\nabla \varphi_i = \begin{bmatrix}
\frac{\partial \varphi_i}{\partial x_0} \\
\frac{\partial \varphi_i}{\partial x_1} \\
\vdots \\
\frac{\partial \varphi_i}{\partial x_{d-1}} \\
\end{bmatrix},
\quad i= 0, 1, \cdots, l-1.
$$


则 $(a(u^0)\nabla\delta u, \nabla v)$ 对应的单元矩阵为 

$$
\boldsymbol A_\tau = 
\int_\tau a(u^0)(\nabla \boldsymbol\varphi)^T\nabla\boldsymbol\varphi
\mathrm d \boldsymbol x
$$

$(a_u'(u^0)\nabla u^0\cdot\delta u, \nabla v)$ 对应的单元矩阵为

$$
\boldsymbol B_\tau = 
\int_\tau 
a_u'(u^0)(\nabla\boldsymbol\varphi)^T\nabla u^0\boldsymbol\varphi
\mathrm d \boldsymbol x
$$

$(f, v)$ 对应的单元列向量为

$$
\boldsymbol b = \int_\tau f\boldsymbol\varphi^T\mathrm d \boldsymbol x
$$

下面讨论边界条件相关的矩阵和向量组装.
设网格边界边（2D)或边界面（3D)上的**局部基函数**个数为
$m$ 个，其组成的**行向量函数**记为

$$
\boldsymbol\omega (\boldsymbol x) = 
\left[
\omega_0(\boldsymbol x), \omega_1(\boldsymbol x), 
\cdots, \omega_{m-1}(\boldsymbol x)
\right]
$$

设 $e$ 是一个边界边或边界面，则 $<\kappa\delta u, v>_e$ 对应的矩阵为

$$
\boldsymbol R_e = 
\int_e 
\kappa \boldsymbol\omega^T\boldsymbol\omega 
\mathrm d \boldsymbol s, \forall e\subset\Gamma_R.
$$

$<g_N, v>_e$  对应的向量为

$$
\boldsymbol b_N = 
\int_e 
g_N\boldsymbol\omega^T
\mathrm d \boldsymbol x, 
\forall e \subset \Gamma_N
$$

$<g_R, v>_e$ 对应的向量为

$$
\boldsymbol b_R = 
\int_e 
g_R\boldsymbol\omega^T
\mathrm d \boldsymbol x, \forall e \subset \Gamma_R 
$$

### 基于 FEALPy 的程序实现

设求解区域为 $\Omega=[0, 1]^2$ 真解设为

$$
u = \cos(\pi x)\cos(\pi y)
$$

非线性扩散系数设为

$$
a(u) = 1 + u^2
$$

边界条件设为纯 Dirichlet 边界条件，其线性化的连续弱形式为

$$
(a(u^0)\nabla\delta u, \nabla v) + (a_u'(u^0)\nabla u^0\cdot\delta u, \nabla v) 
=  (f,v) - (a(u^0)\nabla u^0, \nabla v)
$$

首先导入必要的模块

```python
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
```

接着定义模型数据函数

```python
@cartesian
def solution(p):
    # 真解函数
    pi = np.pi
    x = p[..., 0]
    y = p[..., 1]
    return np.cos(pi*x)*np.cos(pi*y)

@cartesian
def gradient(p):
    # 真解函数的梯度
    x = p[..., 0]
    y = p[..., 1]
    pi = np.pi
    val = np.zeros(p.shape, dtype=np.float64)
    val[...,0] = -pi*np.sin(pi*x)*np.cos(pi*y)
    val[...,1] = -pi*np.cos(pi*x)*np.sin(pi*y)
    return val #val.shape ==p.shape

@cartesian
def source(p):
    # 源项
    x = p[...,0]
    y = p[...,1]
    pi = np.pi
    val = 2*pi**2*(3*np.cos(pi*x)**2*np.cos(pi*y)**2-np.cos(pi*x)**2-np.cos(pi*y)**2+1)*np.cos(pi*x)*np.cos(pi*y)
    return val

@cartesian
def dirichlet(p):
    return solution(p)
```

实现 $(a_u'(u^0)\nabla u^0\cdot\delta u, \nabla v)$ 对应的组装代码

```python
def nolinear_matrix(uh, q=3):

    space = uh.space
    mesh = space.mesh

    qf = mesh.integrator(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    cellmeasure = mesh.entity_measure('cell')

    cval = 2*uh(bcs)[...,None]*uh.grad_value(bcs) # (NQ, NC, GD)
    phi = space.basis(bcs)       # (NQ, 1, ldof)
    gphi = space.grad_basis(bcs) # (NQ, NC, ldof, GD)

    B = np.einsum('q, qcid, qcd, qcj, c->cij', ws, gphi, cval, phi, cellmeasure)

    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:, :, None], shape=B.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)), shape=(gdof,gdof))

    return B
```

准备好求解参数

```python
p = 1 # 有限元空间次数
tol = 1e-8 # 非线性迭代误差限
```

构造 $\Omega=[0, 1]^2$ 上的初始三角形网格，$x$ 和 $y$ 方向都剖分 10 段

```python
domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')
```

构造 `mesh` 上的 $p$ 次 Lagrange 有限元空间 `space`, 并定义一个该空间的函数 `u0` 和 `du`,
其所有自由度系数默认取 0

```python
space = LagrangeFiniteElementSpace(mesh, p=p) # p 的线性元，
u0 = space.function() # 有限元函数 u0 = 0
du = space.function() # 有限元函数 du = 0
```

设置 Dirichlet 边界条件

```python
isDDof = space.set_dirichlet_bc(dirichlet, u0)
isIDof = ~isDDof
```

计算载荷向量

```python
b = space.source_vector(source)
```

定义扩散系数函数

```python
@barycentric
def dcoefficient(bcs):
    return 1+u0(bcs)**2
```

非线性迭代求解

```python
while True:
    A = space.stiff_matrix(c=dcoefficient)
    B = nolinear_matrix(u0)
    U = A + B
    F = b - A@u0
    du[isIDof] = spsolve(U[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
    u0 += du
    err = np.max(np.abs(du))
    print(err)
    if err < tol:
       break
```


第二个是反应项为非线性的例子

$$
-\nabla\left(\nabla u\right) + u^3=f 
$$


满足如下的边界条件：

$$
\begin{aligned}
u =& g_D, \quad\text{on }\Gamma_D\leftarrow \text{Dirichlet } \\
\frac{\partial u}{\partial\boldsymbol n}  =& g_N, 
\quad\text{on }\Gamma_N \leftarrow \text{Neumann} \\
\frac{\partial u}{\partial\boldsymbol n} + \kappa u =& g_R, 
\quad\text{on }\Gamma_R \leftarrow \text{ Robin}
\end{aligned}
$$
