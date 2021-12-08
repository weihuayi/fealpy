---
title: 有限元求解 Poisson 方程
permalink: /docs/zh/start/fem-poisson
key: docs-quick-start-fem-poisson-zh
author: why
---

# Poisson 方程

$\quad$ 给定区域 $\Omega\subset\mathbb R^m$, 其边界 
$\partial \Omega = \Gamma_d \cup \Gamma_n \cup \Gamma_r$.
经典的 Poisson 方程形式如下(方便起见, 称其为 **A 问题**)

$$
\begin{aligned}\tag{A}
    -\Delta u &= f, \quad\text{in }\Omega\\
    u &= g_d, \quad\text{on }\Gamma_d \leftarrow \text{ Dirichlet }\\
    \frac{\partial u}{\partial\boldsymbol n} & = g_n, \quad\text{on
    }\Gamma_n\leftarrow \text{ Neumann}\\
    \frac{\partial u}{\partial\boldsymbol n} + \kappa u& = g_r, \quad\text{on
    }\Gamma_r\leftarrow \text{ Robin}
\end{aligned}
$$

其中 
* 一维情形：$\Delta u(x) = u_{xx}$
* 二维情形：$\Delta u(x, y) = u_{xx} + u_{yy}$
* 三维情形：$\Delta u(x, y, z) = u_{xx} + u_{yy} + u_{zz}$
* $\frac{\partial u}{\partial\boldsymbol n} = \nabla u\cdot\boldsymbol n$, 这里
  $\boldsymbol n$ 是 $\Omega$ 边界的的外法线方向.

$\quad$ 对于绝大多数偏微分方程来说, 往往都找不到**解析形式**的解. 
因此在实际应用问题当中, **数值求解偏微分方程**才是可行的手段.
但要首先解决**偏微分方程方程的无限性**和**计算资源的有限性**这一对本质矛盾.
这里的**无限性**有两个方面的含义, 一个是方程解的无限性, 即要计算出 $\Omega$ 中
**无穷多个点处的函数值**; 另一个是**函数导数**定义的无限性. 
而我们所用的求解工具**计算机**, 从存储和计算速度上来说, 永远都是有限的.
如何克服偏微分方程的无限性, 设计出可以在计算机上高效运行的求解算法,
是**偏微分方程数值解**的主要研究内容.

# 有限元方法简介

$\quad$ 在原来的方程形式下, 要想解决无限性的困难, 
一个可行的办法是**有限差分**方法, 我们会另行讨论. 
这里主要讨论**有限元方法**,  它解决无限性难题的办法是**变分**. 

$\quad$ 要想应用**变分**这一工具, 首先需要引入适当的函数空间, 如

$$
H_{d,0}^1(\Omega) := \{ v\in L^2(\Omega): 
\nabla v \in L^2(\Omega;\mathbb R^m), v|_{\Gamma_d} = 0\},
$$

其中 $L^2(\Omega)$ 是**平方可积**的标量函数组成的空间, $L^2(\Omega;\mathbb R^m)$
表示每个分量都平方可积的 $m$ 维向量函数组成的空间. **注意 $H_{d, 0}^1(\Omega)$ 是无限维的**.

$\quad$ 在 Poisson 方程的两端, 分别乘以任意的 
$v \in H_{d,0}^1(\Omega)$(称其为**测试函数**), 

$$
(f,v) = -(\Delta u, v),
$$

分部积分

$$
\begin{aligned} 
    (f,v)&=-(\Delta u, v)\\
         &=(\nabla u, \nabla v)-<\nabla u \cdot \boldsymbol n,v>_{\partial\Omega}\\
         &=(\nabla u,\nabla v)-<g_N,v>_{\Gamma_n}
         +<\kappa u,v>_{\Gamma_r}-<g_R,v>_{\Gamma_r},
\end{aligned}
$$

整理可得 **A** 问题的**连续弱形式**(称其为 **B 问题**): 寻找 

$$
u\in H^1(\Omega) = \{ v\in L^2(\Omega): \nabla v \in L^2(\Omega;\mathbb R^m)\},
$$

既要满足 Dirichlet 边界 $u\|_{\Gamma_d} = g_d$, 又要满足

$$\tag{B}
(\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_r} = 
(f,v)+<g_r,v>_{\Gamma_r}+<g_n,v>_{\Gamma_n}, \quad\forall v \in H^1_{d, 0}(\Omega),
$$

注意其中 $\Gamma_d$ 上积分项消失了, 原因是测试函数 $v$ 在 Dirichlet 边界上取值为 0. 
上面的推导过程, 把一个逐点意义下成立的方程转化为一个**积分形式的方程**.

$\quad$ 上述的变分过程, 把 **A 问题**变成了 **B 问题**, **其中的动机是什么?**
当然是想把原来无法解决的问题变成一个可以解决的问题, 或者说变成一个更容易解决的问题.
那么 **B 问题**可以求解或者更容易求解了吗? 当然, **B 问题**还是不可以直接求解,
因为空间 $H_{d,0}^1(\Omega)$ 是无限维, 取遍所有的 $v$,
会得到无穷多个不同的积分方程. 但相比于原来的形式, 不再要求**解逐点存在了**, 
方程中的二阶导数也变成了一阶导数, 所以可以说**问题的难度降低了**.

$\quad$ 当然, 把 **A 问题**转化为 **B 问题**, 还有一个重要的理论问题要回答, 即 **B
问题**还和原来的 **A 问题**等价吗? 在一定条件下, 经典的偏微分方程理论可以证明, 
**B 问题**的解存在、唯一, 并且和原问题等价, 这里不再赘述.

$\quad$ 更为重要的是, 方程形式的变化为我们提供了一条从无限走向有限的新途径.
对于**连续弱形式**来说, 核心问题还是出在无限性上, 而解决问题的办法只有一个,
就是**用有限维的空间替代无限维的空间 $H_{d,0}^1(\Omega)$**.

$\quad$ 这里先不讨论有限维空间如何构造(**这是编程要解决的核心问题**), 
后面一系列文章会详细展开. 这里我们先假设有一个 $N$ 维的有限维空间 
$V_N = \text{span}\{ \phi_i \}_0^{N-1}$, 
并把**基函数**组成的向量记为

$$
\boldsymbol \phi = [\phi_0, \phi_1, \cdots, \phi_{N-1}],
$$

这里约定  $\boldsymbol \phi$ 是一个**行向量**. $\boldsymbol \phi$ 的梯度记为

$$
\nabla \boldsymbol \phi = [\nabla \phi_0, \nabla \phi_1, \cdots, \nabla \phi_{N-1}],
$$

这里标量函数的梯度默认是**列向量**的形式. 那么 $\nabla\boldsymbol\phi$
实际上是一个形状为 $N\times d$ 的矩阵函数. 注意,
**这里强调是从基函数的角度看待有限维空间**, **而这种视角转换对编程来说很重要**.

$\quad$ 用 $V_N$ 替代无限维的空间 $H^1_{d,0}(\Omega)$,
并假设要找的解 $u$ 也在这个空间, 重新记为 $u_N$, 满足

$$
u_N = \boldsymbol \phi\boldsymbol u = \sum_{i=0}^{N-1}u_i\phi_i\in V_h,
$$

其中 $\boldsymbol u$ 是 $u_N$ 在基函数 $\boldsymbol\phi$ 下的**坐标向量**, 
或者称为**自由度向量**, 即 

$$
\boldsymbol u =
\begin{bmatrix}
u_0 \\ u_1\\ \vdots \\ u_{N-1}
\end{bmatrix}.
$$

注意这里 $\boldsymbol u$ 是**列向量**. 要指明一下, 空间 $V_N$ 和 $N$ 维欧氏空间 
$\mathbb R^N$ **同构**, 它们中的元素**一一对应**. 
这意味着, 在计算机中只需要存储一个 $N$ 维向量, 
就可以表示一个 $V_N$ 的函数. 进一步, $N$ 维向量的线性运算可以替代 $V_N$ 
中函数之间的线性运算. 另外, 也明确一下, 
这里并不要求 $V_N$ 一定是 $H^1_{d, 0}(\Omega)$ 的子空间.

$\quad$ 进一步可以得到原问题一个新的表达形式, 
即**离散弱形式**(称其为 **C 问题**): 求  $u_N\in V_N$,  既满足 Dirichlet 边界条件 
$u_N|_{\Gamma_d} = g_d$, 又满足

$$\tag{C}
(\nabla u_N,\nabla v_N)+<\kappa u_N, v_N>_{\Gamma_r}= 
(f, v_N)+<g_r, v_N>_{\Gamma_r}+<g_n, v_N>_{\Gamma_n}, 
\quad\forall v_N \in V_N,
$$

表面上 $V_N$ 中仍然有无穷多个 $v_N$, 
但实际上只需要对所有的基函数 $\boldsymbol\phi$ 新的**离散弱形式**成立即可. 
用**矩阵向量**的形式重新改写一下这个**离散弱形式**, 即用
$\boldsymbol\phi\boldsymbol u$ 替换 $u_N$, 基函数向量 
$\boldsymbol \phi$ 替换任意的 $v_N$,
并写成显式积分的形式 

$$
\int_\Omega (\nabla \boldsymbol \phi)^T \nabla\boldsymbol \phi\boldsymbol u
\mathrm d\boldsymbol x +
\int_{\Gamma_r} \kappa\boldsymbol \phi^T \boldsymbol \phi\boldsymbol u
\mathrm d\boldsymbol s = 
\int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x + 
\int_{\Gamma_r} g_r\boldsymbol \phi^T\mathrm d\boldsymbol s + 
\int_{\Gamma_n} g_n\boldsymbol \phi^T\mathrm d\boldsymbol s,
$$

最终可得离散的代数系统

$$
(\boldsymbol A + \boldsymbol R)\boldsymbol u = \boldsymbol b + 
\boldsymbol b_n+ \boldsymbol b_r,
$$

其中

$$
\begin{aligned}
    &\boldsymbol A = \int_\Omega (\nabla \boldsymbol \phi)^T \nabla\boldsymbol \phi\mathrm d\boldsymbol x, \quad 
    \boldsymbol R = \int_{\Gamma_R} \boldsymbol \phi^T \boldsymbol \phi\mathrm d\boldsymbol s, \\
    &\boldsymbol b = \int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x,
    \quad  
    \boldsymbol b_n =  \int_{\Gamma_n} g_n\boldsymbol \phi^T\mathrm d\boldsymbol s,  
    \quad
    \boldsymbol b_r =  \int_{\Gamma_r} g_r\boldsymbol \phi^T\mathrm d\boldsymbol s.
\end{aligned}
$$

只要能组装出上面的矩阵和向量, 原问题最终转化为一个线性代数方程组求解问题,
而该问题是线性代数中已经解决的问题. 当然这里的积分也蕴含无限性的问题(无限求和的极限),
但在被积函数已知的情形下, **数值积分**是解决积分无限性的有力工具.

$\quad$ 当然在理论上, 也同样需要确认 **C 问题**和原来问题是否等价, 
以及它的解是否存在、唯一和稳定, 这是偏微分方程数值解需要研究的重要理论问题. 
显然 **C 问题**和原来的问题已经不等价, 算出的 $u_N$ 不再严格等于 $u$, 
它们之间存在**误差**, 这是用有限替代无限必须付出的代价, 关键是

* 误差有多大?
* 误差可以改善吗?
* 已知数据, 如右端项和边界条件等, 实际应用中都是不精确的, 
  这些不精确的数据又是如何影响解的误差的?

当然我们可以在大多数讲有限元理论的书中找到这些疑问的答案, 这里也不在赘述. 


$\quad$ 回到有限维空间构造的问题, 有限元构造有限维空间的办法, 
是先把求解区域离散为很多简单区域的集合 
$\mathcal T :=\{\tau_i\}_{i=0}^{NC}$, 而这个离散过程就是所谓的**网格生成**. 
这里的 $\tau$ 可以是二维的三角形、
四边形或者更一般的多边形, 三维的四面体、六面体或一般的多面体, 
这样做的好处是可以灵活处理几何形状任意复杂的求解区域, 
这也是有限元能在工业 CAE 仿真中广泛应用的重要原因. 

$\quad$ 进一步, 有限元方法在每个单元 $\tau$ 
上构造局部的多项式基函数, 最后再拼接成全局的基函数. 也就是说, 
最后得到的有限维空间的基函数是分片多项式的, 通常还具有**局部支集性质**
(函数在 $\Omega$ 的一小块子区域上非零, 其它区域上全部为 0). 
这意味着有限元方法最终要计算的矩阵和向量, 
都可转移到每个单元或网格边界边上进行计算, 即:

$$
\begin{aligned}
    \boldsymbol A =& \int_\Omega (\nabla \boldsymbol \phi)^T 
    \nabla\boldsymbol \phi\mathrm d\boldsymbol x 
    = \sum_{\tau\in\mathcal T} \int_\tau (\nabla \boldsymbol \phi|_\tau)^T 
    \nabla\boldsymbol \phi|_\tau\mathrm d\boldsymbol x\\ 
    \boldsymbol R =& \int_{\Gamma_r} \boldsymbol \phi^T \boldsymbol \phi\mathrm d\boldsymbol s 
    = \sum_{e_r\in\Gamma_r}\int_{e_r} (\boldsymbol \phi|_{e_r})^T 
    \boldsymbol \phi|_{e_r}\mathrm d\boldsymbol s\\ 
    \boldsymbol b =& \int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x 
    = \sum_{\tau\in\mathcal T}\int_\tau f (\boldsymbol \phi|_\tau)^T\mathrm d\boldsymbol x \\
    \boldsymbol b_n = & \int_{\Gamma_n} g_N\boldsymbol \phi^T\mathrm d\boldsymbol s 
    = \sum_{e_n\in\Gamma_n}\int_{e_n}g_n(\boldsymbol \phi|_{e_n})^T\mathrm d\boldsymbol s \\
    \boldsymbol b_r = & \int_{\Gamma_r} g_r\boldsymbol \phi^T\mathrm d\boldsymbol s 
    = \sum_{e_r\in\Gamma_r}\int_{e_r}g_r(\boldsymbol \phi|_{e_r})^T\mathrm d\boldsymbol s
\end{aligned}
$$

$\quad$ 在大多数讲有限元的书籍中, 一般都在算法理论
(如存在、唯一、稳定和误差分析等)的讲解上花了过多的功夫, 
力求在数学上的严谨与无懈可击. 严谨对算法理论来说当然很重要, 
但完全忽略算法的实现以及与实际应用的联系, 对计算数学这个学科的发展和应用来说, 
是非常不利的. 所以上面介绍有限元算法的过程, 几乎没有介绍有限元的相关算法理论,
更多是着眼于思想、动机和具体算法的实现. 当然还有更多的算法实现细节没有讲到,
如边界条件处理等, 这些会在后面的文档中逐一介绍.


# FEALPy 求解 Poisson 方程 

给定一个真解为

$$
u  = \cos\pi x\cos\pi y
$$

Poisson 方程, 其定义域为 $[0, 1]^2$, 只有纯的 Dirichlet 边界条件, 下面演示基于
FEALPy 求解这个算例的过程. 

1. 导入创建 pde 模型.
```python
from fealpy.pde.poisson_2d import CosCosData # 导入二维 Poisson 模型实例
pde = CosCosData() # 创建 pde 模型对象
```
2. 生成网格
```python
from fealpy.mesh import MeshFactory as MF # 导入网格工厂模块
domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')
```
3. 建立拉格朗日有限元空间, 代码中 `p=1` 也可以设为更大正整数, 表示建立
$p$ 次元的空间.
```python
# 导入 Lagrange 有限元空间
from fealpy.functionspace import LagrangeFiniteElementSpace 
space = LagrangeFiniteElementSpace(mesh, p=1)  # 线性元空间
```
4. 建立 Dirichlet 边界条件处理对象.
```python
# 导入 Dirichlet 边界处理
from fealpy.boundarycondition import DirichletBC 
bc = DirichletBC(space, pde.dirichlet) # 创建 Dirichlet 边界条件处理对象
```
5. 创建离散代数系统, 并进行边界条件处理. 
```python
uh = space.function() # 创建有限元函数对象
A = space.stiff_matrix() # 组装刚度矩阵对象
F = space.source_vector(pde.source) # 组装右端向量对象
A, F = bc.apply(A, F, uh) # 应用 Dirichlet 边界条件
```
6. 求解离散系统.
```python
# 导入稀疏线性代数系统求解函数
from scipy.sparse.linalg import spsolve
uh[:] = spsolve(A, F)
```
7. 计算 L2 和 H1 误差.
```python
L2Error = space.integralalg.error(pde.solution, uh)
H1Error = space.integralalg.error(pde.gradient, uh.grad_value)
```
7. 画出解函数和网格图像
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
axes = fig.add_subplot(1, 2, 1, projection='3d')
uh.add_plot(axes, cmap='rainbow')
axes = fig.add_subplot(1, 2, 2)
mesh.add_plot(axes)
```

上面是一个典型求解二维 Poisson 方程的例子, 经过简单修改就可以求解 1 维或者 3
维的问题. 更多例子见

1. [带纯 Dirichlet 边界的 Poisson 方程算例.](https://github.com/weihuayi/fealpy/blob/master/example/PoissonFEMWithDirichletBC_example.py)
1. [带纯 Neumann 边界的 Poisson 方程算例.](https://github.com/weihuayi/fealpy/blob/master/example/PoissonFEMWithNeumannBC_example.py)
1. [带纯 Robin 边界的 Poisson 方程算例.](https://github.com/weihuayi/fealpy/blob/master/example/PoissonFEMWithRobinBC_example.py)
1. [带纯 Dirichlet 边界的一般椭圆方程算例.](https://github.com/weihuayi/fealpy/blob/master/example/ConvectinDiffusionReactionFEMwithDirichletBC2d_example.py)
