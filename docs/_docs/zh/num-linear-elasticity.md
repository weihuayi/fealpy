---
title: 线弹性问题数值求解
permalink: /docs/zh/num-linear-elasticity
key: docs-linear-elasticity-zh
---


## PDE 模型

弹性力学研究平衡条件下线弹性问题, 模型共包含三个方程： 

+ 静力平衡方程： $$ -\nabla\cdot \boldsymbol{\sigma} = \boldsymbol{f} $$

+ 几何方程： $$ \boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) $$ 

+ 本构方程： $$ \boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon} + \lambda tr \boldsymbol{\varepsilon}\boldsymbol{I} $$​

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

记位移 $\boldsymbol{u} = \begin{bmatrix} u \\ v \end{bmatrix}$，由几何方程和本构方程可得：

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
& = \boldsymbol{D}\boldsymbol{B}
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
\end{bmatrix}
$$

$$
\boldsymbol{B} = \begin{bmatrix}
\frac{\partial}{\partial x} & 0 \\
0 & \frac{\partial}{\partial y} \\
\frac{\partial}{\partial y} & \frac{\partial}{\partial x}
\end{bmatrix}
$$

称 $\boldsymbol{D}$ 为二维的弹性系数矩阵。记三角形单元 $\tau$ 上的形函数有序集合为 
$\{\varphi_0, \varphi_1, \cdots, \varphi_{n_k-1}\}$，现在我们把位移 
$\boldsymbol{u}$ 的每一个分量用这些函数表示出来，得到：

$$
u = u_0\varphi_0 + u_1\varphi_1 + \cdots + u_{n_k-1}\varphi_{n_k-1}
$$
$$
v = v_0\varphi_0 + v_1\varphi_1 + \cdots + v_{n_k-1}\varphi_{n_k-1}
$$

记 $\boldsymbol{U}$ 为 

$$
\boldsymbol{U} =
\begin{bmatrix}
u_0 \\ u_1 \\ \cdots \\ u_{n_k-1} \\ v_0 \\ v_1 \\ \cdots \\ v_{n_k-1}
\end{bmatrix}
$$

那么

$$
\begin{bmatrix} u \\ v \end{bmatrix} = 
\begin{bmatrix}
\varphi_0 & \varphi_1 & \cdots & \varphi_{n_k-1} & 0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \varphi_0 & \varphi_1 & \cdots & \varphi_{n_k-1}
\end{bmatrix}
\boldsymbol{U}
$$

引入了 $\boldsymbol{D}$ 和 $\boldsymbol{B}$ 后，
$\boldsymbol{\sigma} : \boldsymbol{\varepsilon}$​​ 可以写成如下形式：

$$
\begin{align}
\boldsymbol{\sigma} : \boldsymbol{\varepsilon}  
&= \boldsymbol{D}\boldsymbol{B}
\begin{bmatrix}
u \\ v
\end{bmatrix} : 
\boldsymbol{B}
\begin{bmatrix}
u \\ v
\end{bmatrix} \\
&= \boldsymbol{D}
\begin{bmatrix}
u_x \\ v_y \\ v_x+u_y
\end{bmatrix} : 
\begin{bmatrix}
u_x \\ v_y \\ v_x+u_y
\end{bmatrix} \\
& = \begin{bmatrix}
u_x , v_y , v_x+u_y
\end{bmatrix}
\boldsymbol{D}
\begin{bmatrix}
u_x \\ v_y \\ v_x+u_y
\end{bmatrix}
\end{align}
$$


重新记 $\boldsymbol{B}$ 为

$$
\boldsymbol{B} =
\begin{bmatrix}
\varphi_{0,x} & \varphi_{1,x} & \cdots & \varphi_{n_k-1,x} & 0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \varphi_{0,y} & \varphi_{1,y} & \cdots & \varphi_{n_k-1,y} \\
\varphi_{0,y} & \varphi_{1,y} & \cdots & \varphi_{n_k-1,y} & \varphi_{0,x} 
& \varphi_{1,x} & \cdots & \varphi_{n_k-1,x}
\end{bmatrix}
$$

因此，

$$
\boldsymbol{\sigma} : \boldsymbol{\varepsilon} = 
\boldsymbol{U}^T
\boldsymbol{B}^T \boldsymbol{D}\boldsymbol{B}
\boldsymbol{U}
$$

引入矩阵

$$
\boldsymbol{G} = 
\begin{bmatrix}
\varphi_0 & \varphi_1 & \cdots & \varphi_{n_k-1} & 0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \varphi_0 & \varphi_1 & \cdots & \varphi_{n_k-1}
\end{bmatrix}
$$

那么 $\boldsymbol{f}\cdot\boldsymbol{u}$ 和 $\boldsymbol{g}\cdot\boldsymbol{u}$ 可以分别写成如下形式：

$$
\boldsymbol{f}\cdot\boldsymbol{u} = 
\boldsymbol{U}^T
\boldsymbol{G}^T
\begin{bmatrix}
f_0 \\ f_1
\end{bmatrix}
$$

$$
\boldsymbol{g}\cdot\boldsymbol{u} = 
\boldsymbol{U}^T
\boldsymbol{G}^T
\begin{bmatrix}
g_0 \\ g_1
\end{bmatrix}
$$

引入 $\boldsymbol \varphi_k = [\varphi_0, \varphi_1, \cdots, \varphi_{n_k-1}]$，那么 $\boldsymbol B$ 和 $\boldsymbol G$ 可以分别写成如下形式：

$$
\boldsymbol{B} = 
\begin{bmatrix}
    \frac{\partial \boldsymbol{\varphi}_k}{\partial x} & 0 \\
    0 & \frac{\partial \boldsymbol{\varphi}_k}{\partial y} \\
    \frac{\partial \boldsymbol{\varphi}_k}{\partial y} & \frac{\partial \boldsymbol{\varphi}_k}{\partial x}
\end{bmatrix}
$$

$$
\boldsymbol{G} =
\begin{bmatrix}
    \boldsymbol{\varphi}_k & 0 \\
    0 & \boldsymbol{\varphi}_k
\end{bmatrix}
$$

化简静力平衡方程的变分形式，最终得到：

$$
\int_\tau \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\tau \boldsymbol{G}^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_\tau \boldsymbol{G}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$

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
& = \boldsymbol{D} \boldsymbol{B} 
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
\end{bmatrix}
$$

$$
\boldsymbol{B} = 
\begin{bmatrix}
\frac{\partial }{\partial x} & 0 & 0 \\
0 & \frac{\partial }{\partial y} & 0 \\
0 & 0 & \frac{\partial }{\partial z} \\
0 & \frac{\partial }{\partial z} & \frac{\partial }{\partial y} \\
\frac{\partial }{\partial z} & 0 & \frac{\partial }{\partial x} \\
\frac{\partial }{\partial y} & \frac{\partial }{\partial x} & 0 \\
\end{bmatrix}
$$

称 $\boldsymbol{D}$ 为三维的弹性系数矩阵。记四面体单元 $\tau$ 上的形函数有序集合为 
$\{\varphi_0, \varphi_1, \cdots, \varphi_{n_k-1}\}$，现在我们把位移 
$\boldsymbol{u}$ 的每一个分量用这些函数表示出来，得到：

$$
u = u_0\varphi_0 + u_1\varphi_1 + \cdots + u_{n_k-1}\varphi_{n_k-1}
$$
$$
v = v_0\varphi_0 + v_1\varphi_1 + \cdots + v_{n_k-1}\varphi_{n_k-1}
$$
$$
w = w_0\varphi_0 + w_1\varphi_1 + \cdots + w_{n_k-1}\varphi_{n_k-1}
$$

记 $\boldsymbol{U}$ 为：

$$
\boldsymbol{U} = 
\begin{bmatrix}
u_0 \\ u_1 \\ \cdots \\ u_{n_k-1} \\ 
v_0 \\ v_1 \\ \cdots \\ v_{n_k-1} \\ 
w_0 \\ w_1 \\ \cdots \\ w_{n_k-1}
\end{bmatrix}
$$

那么，

$$
\begin{bmatrix} u \\ v \\ w \end{bmatrix} = 
\begin{bmatrix}
\varphi_0 & \cdots & \varphi_{n_k-1} & 0 & \cdots & 0 & 0 & \cdots & 0 \\
0 & \cdots & 0 & \varphi_0 & \cdots & \varphi_{n_k-1} & 0 & \cdots & 0 \\
0 & \cdots & 0 & 0 & \cdots & 0 & \varphi_0 & \cdots & \varphi_{n_k-1}
\end{bmatrix}
\boldsymbol{U}
$$

同样，引入了 $\boldsymbol{D}$ 和 $\boldsymbol{B}$ 后，
我们可以把 $\boldsymbol{\sigma} : \boldsymbol{\varepsilon}$ 可以写成下面这种形式：

$$
\begin{align*}
\boldsymbol{\sigma} : \boldsymbol{\varepsilon} 
& = \boldsymbol{D}\boldsymbol{B}
\begin{bmatrix}
u \\ v \\ w
\end{bmatrix} : 
\boldsymbol{B}
\begin{bmatrix}
u \\ v \\ w
\end{bmatrix} \\
& = \boldsymbol{D}
\begin{bmatrix}
u_x \\ v_y \\ w_z \\
w_y + v_z \\ w_x + u_z \\ v_x + u_y
\end{bmatrix} : 
\begin{bmatrix}
u_x \\ v_y \\ w_z \\
w_y + v_z \\ w_x + u_z \\ v_x + u_y
\end{bmatrix} \\
& = \begin{bmatrix}
u_x , v_y , w_z , w_y + v_z , w_x + u_z , v_x + u_y
\end{bmatrix}
\boldsymbol{D}
\begin{bmatrix}
u_x \\ v_y \\ w_z \\
w_y + v_z \\ w_x + u_z \\ v_x + u_y
\end{bmatrix}
\end{align*}
$$


重新记 $\boldsymbol{B}$ 为

$$
\boldsymbol{B} =
\begin{bmatrix}
    \frac{\partial \boldsymbol{\varphi}_k}{\partial x} & 0 & 0 \\
    0 & \frac{\partial \boldsymbol{\varphi}_k}{\partial y} & 0 \\
    0 & 0 & \frac{\partial \boldsymbol{\varphi}_k}{\partial z} \\
    0 & \frac{\partial \boldsymbol{\varphi}_k}{\partial z} & \frac{\partial \boldsymbol{\varphi}_k}{\partial y} \\
    \frac{\partial \boldsymbol{\varphi}_k}{\partial z} & 0 & \frac{\partial \boldsymbol{\varphi}_k}{\partial x} \\
    \frac{\partial \boldsymbol{\varphi}_k}{\partial y} & \frac{\partial \boldsymbol{\varphi}_k}{\partial x} & 0
\end{bmatrix}
$$

得到，

$$
\boldsymbol{\sigma} : \boldsymbol{\varepsilon} = 
\boldsymbol{U}^T
\boldsymbol{B}^T \boldsymbol{D}\boldsymbol{B}
\boldsymbol{U}
$$

引入矩阵

$$
\boldsymbol{G} = 
\begin{bmatrix}
    \boldsymbol{\varphi}_k & 0 & 0 \\
    0 & \boldsymbol{\varphi}_k & 0 \\
    0 & 0 & \boldsymbol{\varphi}_k
\end{bmatrix}
$$

那么 $\boldsymbol{f}\cdot\boldsymbol{u}$ 和 $\boldsymbol{g}\cdot\boldsymbol{u}$ 可以分别写成如下形式：

$$
\boldsymbol{f}\cdot\boldsymbol{u} = 
\boldsymbol{U}^T
\boldsymbol{G}^T
\begin{bmatrix}
f_0 \\ f_1 \\ f_1
\end{bmatrix}
$$

$$
\boldsymbol{g}\cdot\boldsymbol{u} = 
\boldsymbol{U}^T
\boldsymbol{G}^T
\begin{bmatrix}
g_0 \\ g_1 \\ g_2
\end{bmatrix}
$$

化简静力平衡方程的变分形式，最终得到：

$$
\int_\tau \boldsymbol{B}^T \boldsymbol{D} \boldsymbol{B} ~ \mathrm{d} \boldsymbol{x} ~ 
\boldsymbol{U}
= \int_\tau \boldsymbol{G}^T \boldsymbol{f} ~ \mathrm{d} \boldsymbol{x}
+ \int_\tau \boldsymbol{G}^T \boldsymbol{g} ~ \mathrm{d} \boldsymbol{x}
$$
