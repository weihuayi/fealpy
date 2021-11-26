---
title: Stokes 方程数值求解
permalink: /docs/zh/example/num-stokes-equation
key: docs-num-stokes-equation-zh
author: wpx
---
# 1. PDE 模型

不可压的流体

$$
\begin{aligned}
- \nabla \cdot \sigma(\boldsymbol u,p) &=  \boldsymbol f  \qquad in \quad \Omega \\
\nabla \cdot \boldsymbol u &= 0 \qquad in \quad \Omega \\
u &= \boldsymbol g \qquad on \quad \partial\Omega\\
\end{aligned}
$$

其中

$$
\sigma(\boldsymbol u ,p) = \mu \varepsilon(\boldsymbol u) + p \boldsymbol I
$$

$$
\varepsilon(\boldsymbol u) = \frac{1}{2} \left(\nabla \boldsymbol u + (\nabla \boldsymbol u)^T\right)
$$

其中符号的物理意义分别为

- $\boldsymbol u$ 代表速度
- $p$ 代表单位面积上的压力
- $\boldsymbol f$ 单位质量流体微团的体积力
- $\mu$ 分子粘性系数

由于不可压缩条件，因此原方程可以变为

$$
\begin{aligned}
- \frac{1}{2} \mu \Delta \boldsymbol u - \nabla p &=  \boldsymbol f  \qquad in \quad \Omega \\
\nabla \cdot \boldsymbol u &= 0 \qquad in \quad \Omega \\
u &= \boldsymbol g \qquad on \quad \partial\Omega\\
\end{aligned}
$$


# 2. 经典变分形式

给定测试向量函数空间 $V$ ，对两边乘上向量测试函数 $\boldsymbol v \in V$ 并在积分区域 $\Omega$ 上做积分

$$
\begin{aligned}
	-\mu \int_{\Omega}\Delta \boldsymbol u \cdot \boldsymbol v \mathrm dx
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	= 
	\int_{\Omega}  \boldsymbol f \cdot \boldsymbol v \mathrm dx
\end{aligned}
$$

对以下几项用散度定理来处理

$$
\begin{aligned}
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) - 
	\nabla \cdot (p\boldsymbol v) \mathrm dx\\
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) dx - 
	\int_{\partial \Omega} p\boldsymbol n \cdot \boldsymbol v \mathrm ds
\end{aligned}
$$

和

$$
\begin{aligned}
	-\int_{\Omega} \Delta \boldsymbol u \cdot \boldsymbol v \quad \mathrm dx 
	&= -\int_{\Omega} \nabla \cdot (\nabla \boldsymbol u \cdot \boldsymbol v) - 
	\nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx\\
	&= -\int_{\partial \Omega} 
	\nabla \boldsymbol u \cdot \boldsymbol v  \cdot \boldsymbol n \mathrm ds + 
	\int_{\Omega} \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx
\end{aligned}
$$

因此其变分形式可以简记为

$$
\begin{aligned}
	-\frac{\mu}{2}\int_{\partial \Omega} 
	\nabla \boldsymbol u \cdot \boldsymbol v  \cdot \boldsymbol n \mathrm ds + 
	\frac{\mu}{2} \int_{\Omega} \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx + 
	\int_{\Omega} p (\nabla \cdot \boldsymbol v) dx - 
	\int_{\partial \Omega} p\boldsymbol n \cdot \boldsymbol v \mathrm ds
	 =  \int_{\Omega} \boldsymbol f \cdot \boldsymbol v \mathrm dx
\end{aligned}
$$

由于具有Dirchlet边界条件，因此测试函数空间在边界上值为零，变分形式变为

$$
\begin{aligned}
	\frac{\mu}{2} \int_{\Omega} 
	\nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx + 
	\int_{\Omega} p (\nabla \cdot \boldsymbol v) dx 
	 =   \int_{\Omega} \boldsymbol f \cdot \boldsymbol v \mathrm dx
\end{aligned}
$$

给定标量测试函数空间 $W$ ，对不可压缩条件两边乘上标量测试函数 $w \in W$ 并在积分区域 $\Omega$ 上做积分
$$
\begin{aligned}
\int_{\Omega} w \nabla \cdot \boldsymbol u &= 0 
\end{aligned}
$$


## 3.二维Lagnrange有限元离散

下面在有限维的 Lagrange 有限元空间中讨论矩阵组装的问题。

记压力p的标量空间的基函数为 

$$
\boldsymbol \psi = [\psi_0, \psi_1, \cdots, \psi_{n_p-1}]
$$

速度 $\boldsymbol u$ 的标量空间的基函数为 

$$
\boldsymbol\phi = [\phi_0, \phi_1, \cdots, \phi_{n_k-1}]
$$

对应向量空间的基为

$$
\boldsymbol\Phi = 
\begin{bmatrix}
\boldsymbol\phi & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\phi
\end{bmatrix}
$$

其梯度形式为

$$
\begin{aligned}
\nabla \boldsymbol \Phi = 
\begin{bmatrix}
	\begin{bmatrix}
		\frac{\partial \phi_0}{\partial x} & 0\\
		 \frac{\partial \phi_0}{\partial y} & 0
	\end{bmatrix}
	& \cdots
	& \begin{bmatrix}
		\frac{\partial \phi_{l-1}}{\partial x} & 0\\
		 \frac{\partial \phi_{l-1}}{\partial y} & 0
	\end{bmatrix}
	& \begin{bmatrix}
		0 & \frac{\partial \phi_{0}}{\partial x} \\
		0 & \frac{\partial \phi_{0}}{\partial y}
		\end{bmatrix}
	& \cdots
	& \begin{bmatrix}
		0 & \frac{\partial \phi_{l-1}}{\partial x} \\
		0 & \frac{\partial \phi_{l-1}}{\partial y}
	\end{bmatrix}
\end{bmatrix}
\end{aligned}
$$

散度形式为

$$
\begin{aligned}
\nabla \cdot \boldsymbol \Phi = 
\begin{bmatrix}
\boldsymbol\phi_x,\boldsymbol\phi_y
\end{bmatrix}
\end{aligned}
$$


设速度有限元解 $\boldsymbol u_h$ 的自由度向量为 $\boldsymbol U$ , 压力有限元解 
$p_h$ 的自由度向量为 $\boldsymbol P$ ,则

$$
\boldsymbol u_h = \boldsymbol\Phi \boldsymbol U , p_h = \boldsymbol \psi \boldsymbol P
$$

注意这里实际上规定了向量 $\boldsymbol U$ 的分量的排列方式，即先排 $x$ 
分量的自由度 $\boldsymbol U_x$，再排 $y$ 分量的自由度 $\boldsymbol U_y$。

则 $\int_{\Omega} \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx$ 
对应的矩阵形式为 

$$
\int_\Omega \boldsymbol (\nabla \boldsymbol \Phi)^T \nabla \boldsymbol \Phi
dx \cdot 
\boldsymbol U = \int_\Omega
\begin{bmatrix}
(\nabla \phi)^T \nabla \phi  & \boldsymbol 0 \\
\boldsymbol 0 & (\nabla \phi^T) \nabla \phi \\ 
\end{bmatrix} 
\mathrm d\boldsymbol x \cdot \boldsymbol U
:=
\begin{bmatrix}
A  & \boldsymbol 0 \\
\boldsymbol 0 & A \\ 
\end{bmatrix} 
\cdot \boldsymbol U
$$

$\int_{\Omega} p (\nabla \cdot \boldsymbol v) \mathrm dx$ 对应的矩阵形式为 

$$
\int_\Omega \boldsymbol (\nabla \cdot \boldsymbol \Phi)^T \boldsymbol \psi
dx \cdot \boldsymbol P 
= 
\int_\Omega
\begin{bmatrix}
\boldsymbol\phi_x^T \boldsymbol \psi \\
\boldsymbol\phi_y^T \boldsymbol \psi \\ 
\end{bmatrix} 
\mathrm d\boldsymbol x \cdot \boldsymbol P
:=
\begin{bmatrix}
B_1 \\
B_2 \\ 
\end{bmatrix} 
\cdot \boldsymbol P
$$

$\int_{\Omega} w \nabla \cdot \boldsymbol u$ 对应的矩阵形式为 

$$
\int_\Omega  \boldsymbol \psi^T \boldsymbol (\nabla \cdot \boldsymbol \Phi)
dx \cdot \boldsymbol U
= 
\int_\Omega
\begin{bmatrix}
\boldsymbol \psi^T \boldsymbol\phi_x & \boldsymbol \psi^T \boldsymbol\phi_y^T 
\end{bmatrix} 
\mathrm d\boldsymbol x \cdot \boldsymbol U
:=
\begin{bmatrix}
\boldsymbol B_1^T , \boldsymbol B_2^T \\ 
\end{bmatrix} 
\cdot \boldsymbol U
$$

因此最终我们右端所要组装的矩阵为

$$
\left[\begin{array}{lll}
\frac{\mu}{2}A & 0 &B_{1} \\
0 & \frac{\mu}{2}A &B_{2}\\
B_{1}^{T} & B_{2}^{T} &0
\end{array}\right]\left[\begin{array}{l}
\boldsymbol{U}_{x} \\
\boldsymbol{U}_{y} \\
\boldsymbol{P}
\end{array}\right]
$$

其中

$$
\begin{aligned}
A &= \int_{\tau}  
\frac{\partial \boldsymbol \phi^T}{\partial x} 
\frac{\partial \boldsymbol \phi}{\partial x} + 
\frac{\partial \boldsymbol \phi^T}{\partial y} 
\frac{\partial \boldsymbol \phi}{\partial y}
\mathrm d \boldsymbol x \\
B_1 &=  \int_{\tau} 
(\frac{\partial \boldsymbol \phi}{\partial x})^T \boldsymbol \psi 
\mathrm d \boldsymbol x \\
B_2 &=  \int_{\tau} 
(\frac{\partial \boldsymbol \phi}{\partial y})^T \boldsymbol \psi 
\mathrm d \boldsymbol x 
\end{aligned}
$$

## 参考文献

