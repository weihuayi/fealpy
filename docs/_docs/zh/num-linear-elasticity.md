---
title: 线弹性问题数值求解
permalink: /docs/zh/num-linear-elasticity
key: docs-linear-elasticity-zh
---


## PDE 模型

弹性力学研究平衡条件下线弹性问题, 模型共包含三个方程： 

+ 静力平衡方程： 
    $$-\nabla\cdot \boldsymbol{\sigma} = \boldsymbol{f}$$
+ 几何方程： 
    $$ \boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) $$ 
+ 本构方程：
    $$ \boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon} + \lambda tr \boldsymbol{\varepsilon}\boldsymbol{I} $$

在实际的工程计算中，这了便于程序实现，$\boldsymbol{\varepsilon}$ 和 $\boldsymbol{\sigma}$ 一般都采用向量化的表示。
静力平衡方程描述了弹性体在平衡条件下应力与体力的关系，几何方程表示应变与位移的关系，本构方程表示应力与应变的关系。 

## 变分形式

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

其中 $ \boldsymbol{g} = \boldsymbol{\sigma}\cdot\boldsymbol{n} $ 为边界 $\partial \Omega_g$ 的边界条件。
因为一个对称张量和一个反对称张量的内积为 $0$。所以 $\nabla\boldsymbol{v}$ 可以分解为一个对
称和一个反对称张量的和，因此上面的变分形式还可以变为

$$
\int_\Omega \boldsymbol{\sigma}(\boldsymbol{u}) : \boldsymbol{\varepsilon}(\boldsymbol{v}) ~ \mathrm{d}\boldsymbol{x} 
= \int_\Omega \boldsymbol{f}\cdot \boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}
+ \int_{\partial \Omega_g} \boldsymbol{g}\cdot\boldsymbol{v} ~ \mathrm{d}\boldsymbol{x}
$$

## 二维情形

记位移函数 $\boldsymbol{u} = \begin{bmatrix} u \\ v \end{bmatrix}$，由张量对象
$\boldsymbol\varepsilon$ 和 $\boldsymbol\sigma$
的对称性，可用降维的方式来表示表示它们。其中 $\boldsymbol\varepsilon$
的降维表示为 

$$
\boldsymbol{\varepsilon} = \begin{bmatrix}
u_x & \frac{v_x + u_y}{2} \\
\frac{v_x + u_y}{2} & v_y \\
\end{bmatrix}
\rightarrow
\begin{bmatrix}
u_x \\ v_y \\\frac{v_x + u_y}{2}
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

可得 $\boldsymbol\sigma$ 的降维表示

$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\sigma}_{11} \\ 
\boldsymbol{\sigma}_{22} \\ 
\boldsymbol{\sigma}_{12}
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
\varepsilon_{11} \\ 
\varepsilon_{22} \\ 
2\varepsilon_{12}
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

矩阵 $\boldsymbol B$ 

$$
\boldsymbol B = 
\begin{bmatrix}
\boldsymbol\Phi_x & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\Phi_y \\
\boldsymbol\Phi_y & \boldsymbol\Phi_x
\end{bmatrix}
$$

进而可得

$$
\boldsymbol B^T \boldsymbol D \boldsymbol B = 
\begin{bmatrix}
\boldsymbol{R_{0,0}} & \boldsymbol{R_{0, 1}} \\
\boldsymbol{R_{1,0}} & \boldsymbol{R_{1, 1}}
\end{bmatrix}
= 
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
(2\mu + \lambda)\boldsymbol\Phi^T_y
\boldsymbol\Phi_y 
+ \mu \boldsymbol\Phi^T_x
\boldsymbol\Phi_x  
\end{bmatrix}
$$

最终可得静力平衡方程的离散矩阵形式

$$
\int_\tau \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\tau \boldsymbol\Psi^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_\tau \boldsymbol{\Psi}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$

注意由上面的推导过程可知，在程序实现过程中，只需要组装好下面三个子矩阵

$$
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_x\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_y\mathrm d\boldsymbol x,\quad
\int_\Omega \boldsymbol\Phi^T_y\boldsymbol\Phi_y\mathrm d\boldsymbol x 
$$

再用拼起来就是最终的刚度矩阵。

## 三维情形

三维情形与二维情形类似，记位移 $\boldsymbol{u} = \begin{bmatrix} u \\ v \\ w \end{bmatrix}$​，
同样由几何方程和本构方程可以得到 $\boldsymbol{\varepsilon}$​ 和 $\boldsymbol{\sigma}$​ 用位移表示的形式。

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

$$
\begin{aligned}
\begin{bmatrix}
\boldsymbol{\sigma}_{11} \\ \boldsymbol{\sigma}_{22} \\ \boldsymbol{\sigma}_{33} \\ 
\boldsymbol{\sigma}_{23} \\ \boldsymbol{\sigma}_{13} \\ \boldsymbol{\sigma}_{12}
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
\varepsilon_{11} \\ \varepsilon_{22} \\ \varepsilon_{33} \\ 
2\varepsilon_{23} \\ 2\varepsilon_{13} \\ 2\varepsilon_{12}
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

矩阵 $\boldsymbol B$ 

$$
\boldsymbol B = 
\begin{bmatrix}
\boldsymbol\Phi_x & \boldsymbol 0 & \boldsymbol 0 \\
\boldsymbol 0 & \boldsymbol\Phi_y & \boldsymbol 0 \\
\boldsymbol 0 & \boldsymbol 0 & \boldsymbol\Phi_z \\
\boldsymbol 0 & \boldsymbol\Phi_z & \boldsymbol\Phi_y\\
\boldsymbol\Phi_z & \boldsymbol 0 & \boldsymbol\Phi_x \\
\boldsymbol\Phi_y & \boldsymbol\Phi_x & \boldsymbol 0 
\end{bmatrix}
$$

进而可得

$$
\begin{aligned}
\boldsymbol B^T \boldsymbol D \boldsymbol B = &
\begin{bmatrix}
\boldsymbol{R_{0,0}} & \boldsymbol{R_{0, 1}} & \boldsymbol{R_{0, 2}}\\
\boldsymbol{R_{1,0}} & \boldsymbol{R_{1, 1}} & \boldsymbol{R_{1, 2}}\\ 
\boldsymbol{R_{2,0}} & \boldsymbol{R_{2, 1}} & \boldsymbol{R_{2, 2}}\\ 
\end{bmatrix} \\
= &  
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
(2\mu + \lambda)\boldsymbol\Phi^T_y
\boldsymbol\Phi_y 
+ \mu \boldsymbol\Phi^T_x
\boldsymbol\Phi_x  
\end{bmatrix}
\end{aligned}
$$

最终可得静力平衡方程的离散矩阵形式

$$
\int_\tau \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\tau \boldsymbol\Psi^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_\tau \boldsymbol{\Psi}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$

注意由上面的推导过程可知，在程序实现过程中，只需要组装好下面三个子矩阵

$$
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_x\mathrm d\boldsymbol x,\quad 
\int_\Omega \boldsymbol\Phi^T_x\boldsymbol\Phi_y\mathrm d\boldsymbol x,\quad
\int_\Omega \boldsymbol\Phi^T_y\boldsymbol\Phi_y\mathrm d\boldsymbol x 
$$

再用拼起来就是最终的刚度矩阵。
