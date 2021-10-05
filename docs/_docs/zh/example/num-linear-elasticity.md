---
title: 线弹性问题数值求解
permalink: /docs/zh/example/num-linear-elasticity
key: docs-linear-elasticity-zh
---


## 1.1 PDE 模型

弹性力学研究平衡条件下线弹性问题, 模型共包含三个方程： 

+ **静力平衡方程**： 

    $$-\nabla\cdot \boldsymbol{\sigma} = \boldsymbol{f}$$

+ **几何方程**： 

    $$ \boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) $$ 

+ **本构方程**：

    $$ \boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon} + \lambda tr \boldsymbol{\varepsilon}\boldsymbol{I} $$

静力平衡方程描述了弹性体在平衡条件下**应力**与**体力**的关系，几何方程表示**应变**与**位移**的关系，本构方程表示**应力**与**应变**的关系。 

在实际的工程计算中，这了便于程序实现和节省计算量，基于 $\boldsymbol{\varepsilon}$ 和 
$\boldsymbol{\sigma}$
都是对称矩阵函数的实事，可以把保它们用一个一维的向量表示，具体细节见下面的推导过程。

## 1.2 经典变分形式

下面我们对静力平衡方程进行变分处理，给定一个向量测试函数空间 $V$, 

$$
-\int_\Omega (\nabla\cdot\boldsymbol{\sigma})\cdot \boldsymbol{v} ~ \mathrm{d} \boldsymbol{x} 
= \int_\Omega \boldsymbol{f}\cdot\boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}, \quad \boldsymbol{v} \in V,
$$

分部积分可得

$$
\int_\Omega \boldsymbol{\sigma} : \nabla\boldsymbol{v} ~ \mathrm{d}\boldsymbol{x} 
= \int_\Omega \boldsymbol{f}\cdot \boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}
+ \int_{\partial \Omega_g} \boldsymbol{g}\cdot\boldsymbol{v} ~ \mathrm{d}\boldsymbol{x},
$$

其中 $\boldsymbol{g} = \boldsymbol{\sigma}\cdot\boldsymbol{n}$ 为边界 
$\partial \Omega_g$ 的边界条件。 因为一个对称张量和一个反对称张量的内积为 $0$。
所以 $\nabla\boldsymbol{v}$ 可以分解为一个对称和一个反对称张量的和，
因此上面的变分形式还可以变为

$$
\int_\Omega \boldsymbol{\sigma}(\boldsymbol{u}) : 
\boldsymbol{\varepsilon}(\boldsymbol{v}) ~ \mathrm{d}\boldsymbol{x} 
= \int_\Omega \boldsymbol{f}\cdot \boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}
+ \int_{\partial \Omega_g} \boldsymbol{g}\cdot\boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}
$$

### 1.2.1 二维 Lagrange 有限元离散

记位移函数 

$$
\boldsymbol{u} = \begin{bmatrix} u \\ v \end{bmatrix},
$$

由张量对象 $\boldsymbol\varepsilon$ 和 $\boldsymbol\sigma$
的对称性，可用向量的形式表示它们。其中 $\boldsymbol\varepsilon$ 的向量表示为 

$$
\boldsymbol{\varepsilon} = 
\begin{bmatrix}
u_x & \frac{v_x + u_y}{2} \\
\frac{v_x + u_y}{2} & v_y \\
\end{bmatrix}
\rightarrow
\begin{bmatrix}
u_x \\ v_y \\ \frac{v_x + u_y}{2}
\end{bmatrix}
$$

由本构方程的展开式

$$
\boldsymbol{\sigma} = 2\mu\begin{bmatrix}
u_x & \frac{v_x + u_y}{2} \\
\frac{v_x + u_y}{2} & v_y \\
\end{bmatrix}
+ \lambda (u_x + v_y)
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
\end{bmatrix}
$$

可得 $\boldsymbol\sigma$ 的向量表示形式

$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\sigma}_{0,0} \\ 
\boldsymbol{\sigma}_{1,1} \\ 
\boldsymbol{\sigma}_{0,1}
\end{bmatrix}
& = \begin{bmatrix}
2\mu u_x + \lambda (u_x + v_y)\\
2\mu v_y + \lambda (u_x + v_y)\\
\mu (v_x + u_y)
\end{bmatrix}\\
& = \begin{bmatrix}
2\mu + \lambda & \lambda & 0\\
\lambda & 2\mu + \lambda & 0\\
0 & 0 & 2\mu
\end{bmatrix}
\begin{bmatrix}
u_x \\ v_y \\ \frac{v_x + u_y}{2}
\end{bmatrix}\\
& = \begin{bmatrix}
2\mu + \lambda & \lambda & 0 \\
\lambda & 2\mu + \lambda & 0 \\
0 & 0 & \mu
\end{bmatrix}
\begin{bmatrix}
\varepsilon_{0,0} \\ 
\varepsilon_{1,1} \\ 
2\varepsilon_{0,1}
\end{bmatrix}\\
& = \boldsymbol{D}\mathcal{B}
\begin{bmatrix}
u \\ v
\end{bmatrix}\\
\end{aligned}
$$

其中，

$$
\boldsymbol{D} = \begin{bmatrix}
2\mu + \lambda & \lambda & 0 \\
\lambda & 2\mu + \lambda & 0 \\
0 & 0 & \mu
\end{bmatrix},\quad 
\mathcal B = \begin{bmatrix}
\frac{\partial}{\partial x} & 0 \\
0 & \frac{\partial}{\partial y} \\
\frac{\partial}{\partial y} & \frac{\partial}{\partial x}
\end{bmatrix}
$$

称 $\boldsymbol{D}$ 为二维的弹性系数矩阵。

下面在有限维的 Lagrange 有限元空间中讨论线弹性矩阵组装的问题。记 Lagrange 标量空间的基函数为 

$$
\boldsymbol\Phi = [\phi_0, \phi_1, \cdots, \phi_{n_k-1}]
$$

对应向量空间的基为

$$
\boldsymbol\Psi = 
\begin{bmatrix}
\boldsymbol\Phi & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\Phi
\end{bmatrix}
$$

设位移有限元解 $\boldsymbol u_h$ 的自由度向量为 $\boldsymbol U$, 则

$$
\boldsymbol u_h = \boldsymbol\Psi\boldsymbol U 
$$

注意这里实际上规定了向量 $\boldsymbol U$ 的分量的排列方式，即先排 $x$
分量的自由度，再排 $y$ 分量的自由度。

矩阵 $\boldsymbol B$ 

$$
\boldsymbol B = \mathcal B\boldsymbol\Psi = 
\begin{bmatrix}
\boldsymbol\Phi_x & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\Phi_y \\
\boldsymbol\Phi_y & \boldsymbol\Phi_x
\end{bmatrix}
$$

进而可得 $\int_\Omega \boldsymbol\sigma(\boldsymbol u_h):\boldsymbol v_h~\mathrm
d\boldsymbol x$ 对应的矩阵形式为 

$$
\begin{aligned}
&\int_\Omega \boldsymbol B^T \boldsymbol D \boldsymbol B ~\mathrm d\boldsymbol x 
\boldsymbol U \\
=& \int_\Omega
\begin{bmatrix}
\boldsymbol{R_{0,0}} & \boldsymbol{R_{0, 1}} \\
\boldsymbol{R_{1,0}} & \boldsymbol{R_{1, 1}}
\end{bmatrix}
~\mathrm d\boldsymbol x 
\boldsymbol U \\
= &\int_\Omega  
\begin{bmatrix}
(2\mu + \lambda)\boldsymbol\Phi^T_x
\boldsymbol\Phi_x 
+ \mu \boldsymbol\Phi^T_y
\boldsymbol\Phi_y & 
\lambda\boldsymbol\Phi^T_x
\boldsymbol\Phi_y 
+ \mu \boldsymbol\Phi^T_y
\boldsymbol\Phi_x \\
\lambda\boldsymbol\Phi^T_y
\boldsymbol\Phi_x 
+ \mu \boldsymbol\Phi^T_x
\boldsymbol\Phi_y & 
 \mu \boldsymbol\Phi^T_x\boldsymbol\Phi_x +
(2\mu + \lambda)\boldsymbol\Phi^T_y
\boldsymbol\Phi_y 
\end{bmatrix}
~\mathrm d\boldsymbol x 
\boldsymbol U 
\end{aligned}
$$

最终可得静力平衡方程的离散矩阵形式

$$
\int_\Omega \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\Omega \boldsymbol\Psi^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_{\partial \Omega_g} \boldsymbol{\Psi}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$

注意由上面的推导过程可知，在程序实现过程中，只需要组装好下面三个子矩阵

$$
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_x\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_y\boldsymbol\Phi_y\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_y\mathrm d\boldsymbol x.
$$

再拼起来就是最终的刚度矩阵。

**注意**, 上面的讨论没有说明 Lagrange
空间次数和所用的网格，因为对任意给的次数，不管三角形或四边形网格，过程都一样的。

### 1.2.2 三维 Lagrange 有限元离散

三维情形与二维情形类似，记位移 $\boldsymbol{u} = \begin{bmatrix} u \\ v \\ w \end{bmatrix}$，
同样由几何方程和本构方程可以得到 $\boldsymbol{\varepsilon}$ 和 $\boldsymbol{\sigma}$ 用位移表示的向量形式。

其中 $\boldsymbol\varepsilon$ 对应的向量形式为

$$
\boldsymbol{\varepsilon} = 
\begin{bmatrix}
u_x & \frac{v_x + u_y}{2} & \frac{w_x + u_z}{2} \\
\frac{v_x + u_y}{2} & v_y & \frac{w_y + v_z}{2} \\
\frac{w_x + u_z}{2} & \frac{w_y + v_z}{2} & w_z
\end{bmatrix}
\rightarrow
\begin{bmatrix}
u_x \\ v_y \\ w_z \\ \frac{w_y + v_z}{2} \\ \frac{w_x + u_z}{2} \\ \frac{v_x + u_y}{2}
\end{bmatrix}
$$

把本构方程展开来写，可得

$$
\boldsymbol{\sigma} =
2\mu\begin{bmatrix}
u_x & \frac{v_x + u_y}{2} & \frac{w_x + u_z}{2} \\
\frac{v_x + u_y}{2} & v_y & \frac{w_y + v_z}{2} \\
\frac{w_x + u_z}{2} & \frac{w_y + v_z}{2} & w_z
\end{bmatrix}
+ \lambda (u_x + v_y + w_z)
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

进而可以得到 $\boldsymbol\sigma$ 的向量展开形式，并写出它与位移函数的关系式

$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\sigma}_{0,0} \\ \boldsymbol{\sigma}_{1,1} \\ \boldsymbol{\sigma}_{2,2} \\ 
\boldsymbol{\sigma}_{1,2} \\ \boldsymbol{\sigma}_{0,2} \\
\boldsymbol{\sigma}_{0,1}
\end{bmatrix}
& = \begin{bmatrix}
2\mu u_x + \lambda (u_x + v_y + w_z) \\
2\mu v_y + \lambda (u_x + v_y + w_z) \\
2\mu w_z + \lambda (u_x + v_y + w_z) \\
\mu (w_y + v_z) \\
\mu (w_x + u_z) \\
\mu (v_x + w_y)
\end{bmatrix} \\
& = \begin{bmatrix}
2\mu + \lambda & \lambda & \lambda & 0 & 0 & 0\\
\lambda & 2\mu + \lambda  & \lambda & 0 & 0 & 0\\
\lambda &  \lambda & 2\mu + \lambda & 0 & 0 & 0 \\
0 & 0 & 0 & 2\mu & 0 & 0 \\
0 & 0 & 0 & 0 & 2\mu & 0 \\
0 & 0 & 0 & 0 & 0 & 2\mu
\end{bmatrix}
\begin{bmatrix}
u_x \\ v_y \\ w_z \\ \frac{w_y + v_z}{2} \\ \frac{w_x + u_z}{2} \\ \frac{v_x + w_y}{2}
\end{bmatrix} \\
& = \begin{bmatrix}
2\mu + \lambda & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & 2\mu + \lambda  & \lambda & 0 & 0 & 0 \\
\lambda &  \lambda & 2\mu + \lambda & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}
\begin{bmatrix}
\varepsilon_{0,0} \\ \varepsilon_{1,1} \\ \varepsilon_{2,2} \\ 
2\varepsilon_{1,2} \\ 2\varepsilon_{0,2} \\ 2\varepsilon_{0,1}
\end{bmatrix}\\
& = \boldsymbol{D} \mathcal B 
\begin{bmatrix}
u \\ v \\ w
\end{bmatrix} \\
\end{aligned}
$$

其中 

$$
\boldsymbol{D} = \begin{bmatrix}
2\mu + \lambda & \lambda & \lambda & 0 & 0 & 0 \\
\lambda & 2\mu + \lambda  & \lambda & 0 & 0 & 0 \\
\lambda &  \lambda & 2\mu + \lambda & 0 & 0 & 0 \\
0 & 0 & 0 & \mu & 0 & 0 \\
0 & 0 & 0 & 0 & \mu & 0 \\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix},\quad
\mathcal B = 
\begin{bmatrix}
\frac{\partial }{\partial x} & 0 & 0 \\
0 & \frac{\partial }{\partial y} & 0 \\
0 & 0 & \frac{\partial }{\partial z} \\
0 & \frac{\partial }{\partial z} & \frac{\partial }{\partial y} \\
\frac{\partial }{\partial z} & 0 & \frac{\partial }{\partial x} \\
\frac{\partial }{\partial y} & \frac{\partial }{\partial x} & 0 \\
\end{bmatrix}
$$

下面在有限维的 Lagrange 有限元空间中讨论线弹性矩阵组装的问题。记 Lagrange 标量空间的基函数为 

$$
\boldsymbol\Phi = [\phi_0, \phi_1, \cdots, \phi_{n_k-1}]
$$

对应三维向量函数空间的基为

$$
\boldsymbol\Psi = 
\begin{bmatrix}
\boldsymbol\Phi & \boldsymbol 0 & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\Phi & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol 0 & \boldsymbol\Phi \\
\end{bmatrix}
$$

设位移有限元解 $\boldsymbol u_h$ 的自由度向量为 $\boldsymbol U$, 则

$$
\boldsymbol u_h = \boldsymbol\Psi\boldsymbol U 
$$

注意这里实际上规定了向量 $\boldsymbol U$ 的分量的排列方式，即先排 $x$
分量的自由度，再排 $y$ 分量的自由度，最后是 $z$ 分量的自由度。

矩阵 $\boldsymbol B$ 

$$
\boldsymbol B = \mathcal B\boldsymbol\Psi = 
\begin{bmatrix}
\boldsymbol\Phi_x & \boldsymbol 0 & \boldsymbol 0 \\
\boldsymbol 0 & \boldsymbol\Phi_y & \boldsymbol 0 \\
\boldsymbol 0 & \boldsymbol 0 & \boldsymbol\Phi_z \\
\boldsymbol 0 & \boldsymbol\Phi_z & \boldsymbol\Phi_y\\
\boldsymbol\Phi_z & \boldsymbol 0 & \boldsymbol\Phi_x \\
\boldsymbol\Phi_y & \boldsymbol\Phi_x & \boldsymbol 0 
\end{bmatrix}
$$

进而可得 $\int_\Omega \boldsymbol\sigma(\boldsymbol u_h):\boldsymbol v_h~\mathrm
d\boldsymbol x$ 对应的矩阵形式为 

$$
\int_\Omega \boldsymbol B^T \boldsymbol D \boldsymbol B \mathrm d\boldsymbol
x \cdot 
\boldsymbol U = \int_\Omega
\begin{bmatrix}
\boldsymbol{R_{0,0}} & \boldsymbol{R_{0, 1}} & \boldsymbol{R_{0, 2}}\\
\boldsymbol{R_{1,0}} & \boldsymbol{R_{1, 1}} & \boldsymbol{R_{1, 2}}\\ 
\boldsymbol{R_{2,0}} & \boldsymbol{R_{2, 1}} & \boldsymbol{R_{2, 2}}\\ 
\end{bmatrix} 
\mathrm d\boldsymbol x \cdot \boldsymbol U
$$

其中

$$
\begin{aligned}
\boldsymbol R_{0, 0} = & (2\mu+\lambda)\boldsymbol\Phi_x^T\boldsymbol\Phi_x +
 \mu\boldsymbol\Phi_y^T\boldsymbol\Phi_y +
 \mu\boldsymbol\Phi_z^T\boldsymbol\Phi_z \\
\boldsymbol R_{1, 1} = & \mu\boldsymbol\Phi_x^T\boldsymbol\Phi_x +
(2\mu+\lambda)\boldsymbol\Phi_y^T\boldsymbol\Phi_y +
\mu\boldsymbol\Phi_z^T\boldsymbol\Phi_z \\
\boldsymbol R_{2, 2} = & \mu\boldsymbol\Phi_x^T\boldsymbol\Phi_x +
\mu\boldsymbol\Phi_y^T\boldsymbol\Phi_y +
(2\mu+\lambda)\boldsymbol\Phi_z^T\boldsymbol\Phi_z \\
\boldsymbol R_{1,2} = & 
\lambda\boldsymbol\Phi^T_y
\boldsymbol\Phi_z 
+ \mu \boldsymbol\Phi^T_z\boldsymbol\Phi_y = \boldsymbol R_{2,1}^T\\
\boldsymbol R_{0,2} = & 
\lambda\boldsymbol\Phi^T_x
\boldsymbol\Phi_z 
+ \mu \boldsymbol\Phi^T_z\boldsymbol\Phi_x = \boldsymbol R_{2, 0}^T\\
\boldsymbol R_{0,1} = &
\lambda\boldsymbol\Phi^T_x
\boldsymbol\Phi_y 
+ \mu \boldsymbol\Phi^T_y
\boldsymbol\Phi_x = \boldsymbol R_{1, 0}^T
\end{aligned}
$$

最终可得静力平衡方程的离散矩阵形式

$$
\int_\Omega \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\Omega \boldsymbol\Psi^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_{\partial\Omega_g} \boldsymbol{\Psi}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$

注意由上面的推导过程可知，在程序实现过程中，只需要组装好下面六个子矩阵

$$
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_x\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_y\boldsymbol\Phi_y\mathrm d\boldsymbol x,\quad
\int_\Omega \boldsymbol\Phi^T_z\boldsymbol\Phi_z\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_z\boldsymbol\Phi_y\mathrm d\boldsymbol x,\\
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_z\mathrm d\boldsymbol x,\quad
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_y\mathrm d\boldsymbol x.
$$

再拼起来就是最终的刚度矩阵。

**注意**, 上面的讨论没有说明 Lagrange
空间次数和所用的网格，因为对任意给的次数，不管四面体、六面体或者三棱柱等网格，
过程都一样的。


## 1.3 混合变分形式

对本构方程变形, 可得 

$$
\frac{1}{2\mu}\left(\boldsymbol{\sigma}-\frac{tr\boldsymbol{\sigma}}{2\mu+\lambda d}\boldsymbol{I}\right)-\boldsymbol{\varepsilon} =0 \tag{27}
$$

记 

$$
\mathcal{A}(\boldsymbol{\sigma})=
\frac{1}{2\mu}\left(
\boldsymbol{\sigma}-\frac{tr\boldsymbol{\sigma}}{2\mu+\lambda d}\boldsymbol{I}
\right)
$$

对于 $\boldsymbol{\tau}\in H(\text{div};\mathbb{S})$，上式与$\boldsymbol{\tau}$做内积，并分部积分得

$$
\int_{\Omega}\mathcal{A}(\boldsymbol{\sigma}):\boldsymbol{\tau}~d\boldsymbol{x}+\int_{\Omega}(\nabla\cdot\boldsymbol{\tau})\cdot\boldsymbol{u}~d\boldsymbol{x} = \int_{\partial\Omega}(\boldsymbol{\tau}\cdot{\boldsymbol{n}})\cdot\boldsymbol{u}~d\boldsymbol{s},
$$

结合$(4)$,就有混合变分形式为: 

$$
\begin{cases}
\text{Find }(\boldsymbol{\sigma},\boldsymbol{u})\in H(\text{div},\Omega,\mathbb{S})\times(L^2(\Omega))^d, \text{ such that}\\
(\mathcal{A}(\boldsymbol{\sigma}),\boldsymbol{\tau})_{\Omega}+(\text{div}~\boldsymbol{\tau},\boldsymbol{u}) = (\boldsymbol{\tau}\cdot\boldsymbol{n},\boldsymbol{u})_{\partial\Omega},&\text{for all } \boldsymbol{\tau}\in H(\text{div},\Omega,\mathbb{S}),\\
(\text{div}\boldsymbol{\sigma},\boldsymbol{v})_{\Omega} = (-\boldsymbol{f},\boldsymbol{v})_{\Omega},&\text{for all }\boldsymbol{v}\in(L^2(\Omega))^d.
\end{aligned} 
$$

注：上述对应的是纯位移的边界条件的混合变分形式，对于一般含有应力的边界条件

$$
\boldsymbol{g}_n=\boldsymbol{\sigma}\cdot\boldsymbol{n} 
$$
on $\partial\Omega_{\boldsymbol{g}_n}$, $\boldsymbol{g}_d=\boldsymbol{u}$ on $\partial\Omega_{\boldsymbol{g}_d}$， 对应的混合变分形式为:
$$
\begin{aligned}
\begin{cases}
\text{Find }(\boldsymbol{\sigma},\boldsymbol{u})\in \Sigma_{\boldsymbol{g}_n}\times U, \text{ such that}\\
(\mathcal{A}(\boldsymbol{\sigma}),\boldsymbol{\tau})_{\Omega}+(\text{div}~\boldsymbol{\tau},\boldsymbol{u}) = (\boldsymbol{\tau}\cdot\boldsymbol{n},\boldsymbol{g}_d)_{\partial\Omega_{\boldsymbol{g}_d}}&\text{for all } \boldsymbol{\tau}\in \Sigma_{\boldsymbol{0}},\\
(\text{div}\boldsymbol{\sigma},\boldsymbol{v})_{\Omega} = (-\boldsymbol{f},\boldsymbol{v})_{\Omega},&\text{for all }\boldsymbol{v}\in U.
\end{cases}
\end{aligned}
$$
其中 $\Sigma_{\boldsymbol{\psi}}=\{\boldsymbol{\tau}\in H(\text{div},\Omega,\mathbb{S}):~\boldsymbol{\tau}\cdot\boldsymbol{n}=\boldsymbol{\psi} \text{ on } \partial\Omega_{\boldsymbol{g}_n}\},~ U=(L^2(\Omega))^d$.

### 1.3.1 二维胡张有限元离散

由应力函数 $\boldsymbol{\sigma}$ 有对称性，其向量表示为

$$
\boldsymbol{\sigma}=
\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}&\boldsymbol{\sigma}_{0,1}\\
\boldsymbol{\sigma}_{0,1}&\boldsymbol{\sigma}_{1,1}\\
\end{bmatrix}\to\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}\\
\boldsymbol{\sigma}_{1,1}\\
\boldsymbol{\sigma}_{0,1}
\end{bmatrix},
$$

这样 $\text{div}\boldsymbol{\sigma}$ 为

$$
\begin{aligned}
\text{div}\boldsymbol{\sigma}&=
\begin{bmatrix}
\partial_{x}\boldsymbol{\sigma}_{0,0}+\partial_{y}\boldsymbol{\sigma}_{0,1}\\
\partial_{x}\boldsymbol{\sigma}_{0,1}+\partial_{y}\boldsymbol{\sigma}_{1,1}
\end{bmatrix}
\\
&=\boldsymbol{\mathcal{B}}^T\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}\\
\boldsymbol{\sigma}_{1,1}\\
\boldsymbol{\sigma}_{0,1}
\end{bmatrix},
\end{aligned}
$$
其中 $\boldsymbol{\mathcal{B}}$ 的定义见$(10)$.

由$(27)$得到 $\boldsymbol{\varepsilon}$ 的向量表示形式
$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\varepsilon}_{0,0}\\
\boldsymbol{\varepsilon}_{1,1}\\
2\boldsymbol{\varepsilon}_{0,1}\\
\end{bmatrix}
&=
\begin{bmatrix}
\frac{1}{2\mu}\left[\boldsymbol{\sigma}_{0,0}-(\boldsymbol{\sigma}_{0,0}+\boldsymbol{\sigma}_{1,1})/(2\mu+\lambda d)\right]\\
\frac{1}{2\mu}\left[\boldsymbol{\sigma}_{1,1}-(\boldsymbol{\sigma}_{0,0}+\boldsymbol{\sigma}_{1,1})/(2\mu+\lambda d)\right]\\
\frac{1}{\mu}\boldsymbol{\sigma}_{0,1}
\end{bmatrix}\\
&=\frac{1}{2\mu}\boldsymbol{A}
\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}\\
\boldsymbol{\sigma}_{1,1}\\
\boldsymbol{\sigma}_{0,1}
\end{bmatrix},
\end{aligned}
$$
其中，
$$
\boldsymbol{A} = \begin{bmatrix}
1-1/(2\mu+\lambda d)&-1/(2\mu+\lambda d)&0\\
-1/(2\mu+\lambda d)&1-1/(2\mu+\lambda d)&0\\
0&0&2
\end{bmatrix}.
$$

在有限维胡张元空间中讨论线弹性矩阵组装问题，记胡张元空间基函数为
$$
\boldsymbol{\Sigma}=
\begin{bmatrix}
\boldsymbol{\varphi}_0,\boldsymbol{\varphi}_1,\cdots,\boldsymbol{\varphi}_{N_k-1}
\end{bmatrix},
$$

其中 

$$\boldsymbol{\varphi}_i=[\boldsymbol{\varphi}_{i,0},\boldsymbol{\varphi}_{i,1},\boldsymbol{\varphi}_{i,2}]^T,$$ 

分别对应

$$[\boldsymbol{\sigma}_{0,0},\boldsymbol{\sigma}_{1,1},\boldsymbol{\sigma}_{0,1}]^T.$$

 设应力有限元解的自由度向量为 $\boldsymbol{X}$，则
$$\begin{aligned}
\boldsymbol{\sigma}_h&=\boldsymbol{\Sigma}\boldsymbol{X},\\
\text{div}\boldsymbol{\sigma}_h&=\boldsymbol{\mathcal{B}}^T\boldsymbol{\Sigma}\boldsymbol{X},
\end{aligned}$$
令矩阵 $\boldsymbol{B}$
$$
\begin{aligned}
\boldsymbol{B}&=\boldsymbol{\mathcal{B}}^T\boldsymbol{\Sigma}\\
&=
\begin{bmatrix}\partial_{x}\boldsymbol{\Sigma_{0}}+\partial_{y}\boldsymbol{\Sigma_{2}}\\
\partial_{y}\boldsymbol{\Sigma_{1}}+\partial_{x}\boldsymbol{\Sigma_{2}}
\end{bmatrix}\\
&=
\begin{bmatrix}
\boldsymbol{B}_0\\
\boldsymbol{B}_1
\end{bmatrix},
\end{aligned}
$$

其中 $\boldsymbol{\Sigma}_i,\boldsymbol{B}_i$ 为 $\boldsymbol{\Sigma}, \boldsymbol{B}$ 的第 $i$ 行.

位移有限元解表达式同 $(13)$, 不过 $\boldsymbol{\Phi}$ 为间断标量空间基函数.


进而 $(\boldsymbol{\mathcal{A}}(\boldsymbol{\sigma}_h),\boldsymbol{\tau}_h)_{\Omega}$ 对应的矩阵形式为

$$
\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x}~\boldsymbol{X},
$$

$(\text{div}\boldsymbol{\sigma}_h,\boldsymbol{v}_h)_{\Omega}$ 对应的矩阵形式为

$$
\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}~\boldsymbol{X}
=
\int_{\Omega}\begin{bmatrix}
\boldsymbol{\Phi}^T\boldsymbol{B}_0\\
\boldsymbol{\Phi}^T\boldsymbol{B}_1
\end{bmatrix}~d\boldsymbol{x}~\boldsymbol{X},
$$

$(\text{div}\boldsymbol{\tau}_h,\boldsymbol{u}_h)_{\Omega}$ 对应的矩阵形式为

$$
\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)^T~\boldsymbol{U}.
$$

最终得到本构方程和静力平衡方程的离散矩阵形式

$$
\begin{bmatrix}
\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x}&\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)^T\\
\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)&\boldsymbol{0}
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{X}\\
\\
\boldsymbol{U}
\end{bmatrix}
=
\begin{bmatrix}
\int_{\partial\Omega_{\boldsymbol{g}_d}}\left(\boldsymbol{\Sigma}^T\boldsymbol{C}\boldsymbol{n}\right)\boldsymbol{g}_d~d\boldsymbol{s}\\
-\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{f}~d\boldsymbol{x}
\end{bmatrix},
$$

其中

$$\boldsymbol{C}=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
$$
由上面的推导过程可知，在程序实现过程中，只需要组装好下面三个子矩阵

$$\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x},~\int_{\Omega}\boldsymbol{\Phi}^T\boldsymbol{B}_0~d\boldsymbol{x},~\int_{\Omega}\boldsymbol{\Phi}^T\boldsymbol{B}_1~d\boldsymbol{x}$$

再拼起来就是最终的刚度矩阵。

### 1.3.2 三维胡张有限元离散

三维情形与二维情形类似，由应力函数 $\boldsymbol{\sigma}$ 有对称性，其向量表示为
$$
\boldsymbol{\sigma}=
\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}&\boldsymbol{\sigma}_{0,1}&\boldsymbol{\sigma}_{0,2}\\
\boldsymbol{\sigma}_{0,1}&\boldsymbol{\sigma}_{1,1}&\boldsymbol{\sigma}_{1,2}\\
\boldsymbol{\sigma}_{0,2}&\boldsymbol{\sigma}_{1,2}&\boldsymbol{\sigma}_{2,2}\\
\end{bmatrix}\to\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}\\
\boldsymbol{\sigma}_{1,1}\\
\boldsymbol{\sigma}_{2,2}\\
\boldsymbol{\sigma}_{1,2}\\
\boldsymbol{\sigma}_{0,2}\\
\boldsymbol{\sigma}_{0,1}\\
\end{bmatrix},
$$
这样 $\text{div}\boldsymbol{\sigma}$ 为
$$
\begin{aligned}
\text{div}\boldsymbol{\sigma}&=
\begin{bmatrix}
\partial_{x}\boldsymbol{\sigma}_{0,0}+\partial_{y}\boldsymbol{\sigma}_{0,1}+\partial_{z}\boldsymbol{\sigma}_{0,2}\\
\partial_{x}\boldsymbol{\sigma}_{0,1}+\partial_{y}\boldsymbol{\sigma}_{1,1}+\partial_{z}\boldsymbol{\sigma}_{1,2}\\
\partial_{x}\boldsymbol{\sigma}_{0,2}+\partial_{y}\boldsymbol{\sigma}_{1,2}+\partial_{z}\boldsymbol{\sigma}_{2,2}
\end{bmatrix}
\\
&=\boldsymbol{\mathcal{B}}^T\begin{bmatrix}
\boldsymbol{\sigma}_{0,0}\\
\boldsymbol{\sigma}_{1,1}\\
\boldsymbol{\sigma}_{2,2}\\
\boldsymbol{\sigma}_{1,2}\\
\boldsymbol{\sigma}_{0,2}\\
\boldsymbol{\sigma}_{0,1}\\
\end{bmatrix},
\end{aligned}
$$
其中 $\boldsymbol{\mathcal{B}}$ 的定义见 $(19)$.

由 $(27)$ 得到 $\boldsymbol{\varepsilon}$ 的向量表示形式
$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\varepsilon}_{0,0}\\
\boldsymbol{\varepsilon}_{1,1}\\
\boldsymbol{\varepsilon}_{2,2}\\
2\boldsymbol{\varepsilon}_{1,2}\\
2\boldsymbol{\varepsilon}_{0,2}\\
2\boldsymbol{\varepsilon}_{0,1}\\
\end{bmatrix}
&=
\begin{bmatrix}
\frac{1}{2\mu}\left[\boldsymbol{\sigma}_{0,0}-(\boldsymbol{\sigma}_{0,0}+\boldsymbol{\sigma}_{1,1}+\boldsymbol{\sigma}_{2,2})/(2\mu+\lambda d)\right]\\
\frac{1}{2\mu}\left[\boldsymbol{\sigma}_{1,1}-(\boldsymbol{\sigma}_{0,0}+\boldsymbol{\sigma}_{1,1}+\boldsymbol{\sigma}_{2,2})/(2\mu+\lambda d)\right]\\
\frac{1}{2\mu}\left[\boldsymbol{\sigma}_{2,2}-(\boldsymbol{\sigma}_{0,0}+\boldsymbol{\sigma}_{1,1}+\boldsymbol{\sigma}_{2,2})/(2\mu+\lambda d)\right]\\
\frac{1}{\mu}\boldsymbol{\sigma}_{1,2}\\
\frac{1}{\mu}\boldsymbol{\sigma}_{0,2}\\
\frac{1}{\mu}\boldsymbol{\sigma}_{0,1}\\
\end{bmatrix}\\
&=\frac{1}{2\mu}\boldsymbol{A}
\begin{bmatrix}
\boldsymbol{\varepsilon}_{0,0}\\
\boldsymbol{\varepsilon}_{1,1}\\
\boldsymbol{\varepsilon}_{2,2}\\
2\boldsymbol{\varepsilon}_{1,2}\\
2\boldsymbol{\varepsilon}_{0,2}\\
2\boldsymbol{\varepsilon}_{0,1}\\
\end{bmatrix},
\end{aligned}
$$
其中，
$$
\boldsymbol{A} = \begin{bmatrix}
1-1/(2\mu+\lambda d)&-1/(2\mu+\lambda d)&-1/(2\mu+\lambda d)&0&0&0\\
-1/(2\mu+\lambda d)&1-1/(2\mu+\lambda d)&-1/(2\mu+\lambda d)&0&0&0\\
-1/(2\mu+\lambda d)&-1/(2\mu+\lambda d)&1-1/(2\mu+\lambda d)&0&0&0\\
0&0&0&2&0&0\\
0&0&0&0&2&0\\
0&0&0&0&0&2\\
\end{bmatrix}.
$$

在有限维胡张元空间中讨论线弹性矩阵组装问题，记胡张元空间基函数为
$$
\boldsymbol{\Sigma}=
\begin{bmatrix}
\boldsymbol{\varphi}_0,\boldsymbol{\varphi}_1,\cdots,\boldsymbol{\varphi}_{N_k-1}
\end{bmatrix},
$$
其中
$$\boldsymbol{\varphi}_i=[\boldsymbol{\varphi}_{i,0},\boldsymbol{\varphi}_{i,1},\boldsymbol{\varphi}_{i,2},\boldsymbol{\varphi}_{i,3},\boldsymbol{\varphi}_{i,4},\boldsymbol{\varphi}_{i,5}]^T,$$ 

分别对应 

$$[\boldsymbol{\sigma}_{0,0},\boldsymbol{\sigma}_{1,1},\boldsymbol{\sigma}_{2,2},\boldsymbol{\sigma}_{1,2},\boldsymbol{\sigma}_{0,2},\boldsymbol{\sigma}_{0,2}]^T.$$

设应力有限元解的自由度向量为 $\boldsymbol{X}$，则
$$
\begin{aligned}
\boldsymbol{\sigma}_h&=\boldsymbol{\Sigma}\boldsymbol{X},\\
\text{div}\boldsymbol{\sigma}_h&=\boldsymbol{\mathcal{B}}^T\boldsymbol{\Sigma}\boldsymbol{X}\\
\end{aligned}
$$
令矩阵 $\boldsymbol{B}$
$$
\begin{aligned}
\boldsymbol{B}&=\boldsymbol{\mathcal{B}}^T\boldsymbol{\Sigma}\\
&=\begin{bmatrix}
\partial_x\boldsymbol{\Sigma}_0+\partial_y\boldsymbol{\Sigma}_5+\partial_z\boldsymbol{\Sigma}_4\\
\partial_x\boldsymbol{\Sigma}_5+\partial_y\boldsymbol{\Sigma}_1+\partial_z\boldsymbol{\Sigma}_3\\
\partial_x\boldsymbol{\Sigma}_4+\partial_y\boldsymbol{\Sigma}_3+\partial_z\boldsymbol{\Sigma}_2
\end{bmatrix}
\boldsymbol{X}\\
&=\begin{bmatrix}
\boldsymbol{B}_0\\
\boldsymbol{B}_1\\
\boldsymbol{B}_2
\end{bmatrix},
\end{aligned}
$$
其中 $\boldsymbol{\Sigma}_i,\boldsymbol{B}_i$ 为 $\boldsymbol{\Sigma}, \boldsymbol{B}$ 的第 $i$ 行.

位移有限元解表达式同 $(22)$ , 不过$\boldsymbol{\Phi}$为间断标量空间基函数.


进而可得 $(\boldsymbol{\mathcal{A}}(\boldsymbol{\sigma}_h),\boldsymbol{\tau}_h)_{\Omega}$ 对应的矩阵形式为
$$
\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x}~\boldsymbol{X}
$$
$(\text{div}\boldsymbol{\sigma}_h,\boldsymbol{v}_h)_{\Omega}$ 对应的矩阵形式为
$$
\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}~\boldsymbol{X}=\int_{\Omega}\begin{bmatrix}
\boldsymbol{\Phi}^T\boldsymbol{B}_0\\
\boldsymbol{\Phi}^T\boldsymbol{B}_1\\
\boldsymbol{\Phi}^T\boldsymbol{B}_2
\end{bmatrix}~d\boldsymbol{x}~\boldsymbol{X} 
$$
$(\text{div}\boldsymbol{\tau}_h,\boldsymbol{u}_h)_{\Omega}$对应的矩阵形式为
$$
\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)^T~\boldsymbol{U}.
$$
最终得到本构方程和静力平衡方程的离散矩阵形式
$$
\begin{bmatrix}
\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x}&\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)^T\\
\left(\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{B}~d\boldsymbol{x}\right)&\boldsymbol{0}
\end{bmatrix}
\begin{bmatrix}
\boldsymbol{X}\\
\\
\boldsymbol{U}
\end{bmatrix}
=
\begin{bmatrix}
\int_{\partial\Omega_{\boldsymbol{g}_d}}\left(\boldsymbol{\Sigma}^T\boldsymbol{C}\boldsymbol{n}\right)\boldsymbol{g}_d~d\boldsymbol{s}\\
-\int_{\Omega}\boldsymbol{\Psi}^T\boldsymbol{f}~d\boldsymbol{x}
\end{bmatrix},
$$
其中
$$\boldsymbol{C}=
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1\\
0&1&1\\
1&0&1\\
1&1&0
\end{bmatrix}.
$$

由上面的推导过程可知，在程序实现过程中，只需要组装好下面四个子矩阵

$$\frac{1}{2\mu}\int_{\Omega}\boldsymbol{\Sigma}^T\boldsymbol{A}\boldsymbol{\Sigma}~d\boldsymbol{x},~\int_{\Omega}\boldsymbol{\Phi}^T\boldsymbol{B}_0~d\boldsymbol{x},~\int_{\Omega}\boldsymbol{\Phi}^T\boldsymbol{B}_1~d\boldsymbol{x},~\int_{\Omega}\boldsymbol{\Phi}^T\boldsymbol{B}_2~d\boldsymbol{x},$$

再拼起来就是最终的刚度矩阵。
