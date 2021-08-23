---
title: Navier-stokes 方程数值求解
permalink: /docs/zh/num-navier-stokes-equation
key: docs-num-navier-stoke-equation-zh
---

# Navier-stokes 问题

## PDE 模型

不可压的流体

$$
\begin{cases}
\rho (\frac{\partial \bm u}{\partial t}+\bm u \cdot \nabla\bm u)  =
　-\nabla p + \nabla \cdot \sigma(\bm u) +\rho \bm f \\
\nabla \cdot \bm u = 0
\end{cases}
$$

若其为牛顿流体的话，流体的切应力与应变时间(速度梯度)成正比。其关系如下

$$
\sigma(\bm u) = 2 \mu \varepsilon(\bm u)
$$

$$
\varepsilon(\bm u) = \frac{1}{2} (\nabla \bm u + (\nabla \bm u)^T)
$$

其中符号的物理意义分别为

- $\bm u$ 代表速度
- $p$ 代表单位面积上的压力
- $f$ 单位质量流体微团的体积力
- $\mu$ 分子粘性系数

#　变分格式

###### 对两便乘上向量测试函数 $\bm v \in V$ 并在积分区域 $\Omega$ 上做积分

$$
\begin{eqnarray}
	\int_{\Omega} \rho \frac{\partial \boldsymbol u}{\partial t}\boldsymbol v
    \mathrm dx
	+ \int_{\Omega} \rho \boldsymbol u \cdot \nabla \boldsymbol u \cdot \boldsymbol v \mathrm dx 
	= 
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	+\mu \int_{\Omega}(\nabla \cdot (\nabla \boldsymbol u + \nabla (\boldsymbol u)^T)) \cdot \boldsymbol v \mathrm dx
	+\int_{\Omega} \rho \boldsymbol f \cdot \boldsymbol v \mathrm dx
\end{eqnarray}
$$

对以下几项用散度定理来处理

$$
\begin{align}
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) - \nabla \cdot (p\boldsymbol v) \mathrm dx\\
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) dx - \int_{\partial \Omega} p\boldsymbol v \cdot \boldsymbol n \mathrm ds
\end{align}
$$

和

$$
\begin{align}
	\int_{\Omega} (\nabla \cdot \nabla \boldsymbol u) \cdot \boldsymbol v \quad \mathrm dx 
	&= \int_{\Omega} \nabla \cdot (\nabla \boldsymbol u \cdot \boldsymbol v) - \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx\\
	&= \int_{\partial \Omega} \nabla \boldsymbol u \cdot \boldsymbol v  \cdot \boldsymbol n \mathrm ds - \int_{\Omega} \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx
\end{align}
$$

因此我们可得到其变分形式

$$
\begin{align}
	(\rho \frac{\partial \boldsymbol u}{\partial t},\boldsymbol v) + (\rho \boldsymbol u \cdot \nabla \boldsymbol u ,\boldsymbol v ) 
	- ( p ,\nabla \cdot \boldsymbol v) + (p\boldsymbol n ,\boldsymbol v)_{\partial \Omega} \\
	+( \sigma (\boldsymbol u  + (\boldsymbol u)^T) , \nabla \boldsymbol v) 
	-( \sigma (\boldsymbol u + (\boldsymbol u)^T) \cdot \boldsymbol n ,  \boldsymbol v))_{\partial \Omega}
	 =  (\rho \boldsymbol f,\boldsymbol v)
\end{align}
$$

由于任何一个矩阵都可以分解成一个对称矩阵和反对称矩阵的求和，即

$$
\nabla \boldsymbol v = \frac{\nabla \boldsymbol v + (\nabla \boldsymbol v)^T}{2} + \frac{\nabla \boldsymbol v - (\nabla\boldsymbol v)^T}{2}
$$

易证反对陈矩阵和对称矩阵求内积会消失，所以变分形式可以变为

$$
\begin{align}
	(\rho \frac{\partial \boldsymbol u}{\partial t},\boldsymbol v) + (\rho \boldsymbol u \cdot \nabla \boldsymbol u ,\boldsymbol v ) 
	- ( p ,\nabla \cdot \boldsymbol v) + (p\boldsymbol n ,\boldsymbol v)_{\partial \Omega} \\
	+( \sigma(\boldsymbol u) , \varepsilon(\boldsymbol v)) 
	-( \sigma(\boldsymbol u) \cdot \boldsymbol n ,  \boldsymbol v))_{\partial \Omega}
	=  (\rho \boldsymbol f,\boldsymbol v)
\end{align}
$$

## IPCS方法

给定 $\Omega$​ 上一个单纯形网格离散 $\tau$, 构造连续的分p 次$Lgarange$多项式空间, 其基函数向量记为：

$$
\bm\phi := [\phi_0(\bm x), \phi_1(\bm x), \cdots, \phi_{l-1}(\bm x)],
\forall \bm x \in \tau
$$


$$
\nabla \bm \phi = 
\begin{bmatrix}
	\frac{\partial \phi_0}{\partial x} & \frac{\partial \phi_1}{\partial x} & \cdots \frac{\partial \phi_{l-1}}{\partial x} \\
	\frac{\partial \phi_0}{\partial y} & \frac{\partial \phi_1}{\partial y} & \cdots \frac{\partial \phi_{l-1}}{\partial y}
\end{bmatrix}
:=\begin{bmatrix}
	\frac{\partial \bm \phi}{\partial x} \\
	\frac{\partial \bm \phi}{\partial y}
\end{bmatrix}
$$
则其对应的质量矩阵为
$$
\begin{equation}
	\bm H=\int_{\tau} \bm \phi^{T} \bm \phi d \bm x=
	\begin{bmatrix}
		(\phi_{0}, \phi_{0})_{\tau} & (\phi_{0}, \phi_{1})_{\tau} & \cdots & (\phi_{0}, \phi_{l-1})_{\tau} \\
		(\phi_{1}, \phi_{0})_{\tau} & (\phi_{1}, \phi_{1})_{\tau} & \cdots & (\phi_{1}, \phi_{l-1})_{\tau} \\
		\vdots & \vdots & \ddots & \vdots \\
		(\phi_{l-1}, \phi_{0})_{\tau} & (\phi_{l-1}, \phi_{1})_{\tau} & \cdots & (\phi_{l-1}, \phi_{l-1})_{\tau}
	\end{bmatrix}
\end{equation}
$$
$$
\begin{equation}
	\bm G=\int_{\tau} \bm \phi^{T} \bm \phi d \bm x=
	\begin{bmatrix}
		(\phi_{0}, \phi_{0})_{\tau} & (\phi_{0}, \phi_{1})_{\tau} & \cdots & (\phi_{0}, \phi_{l-1})_{\tau} \\
		(\phi_{1}, \phi_{0})_{\tau} & (\phi_{1}, \phi_{1})_{\tau} & \cdots & (\phi_{1}, \phi_{l-1})_{\tau} \\
		\vdots & \vdots & \ddots & \vdots \\
		(\phi_{l-1}, \phi_{0})_{\tau} & (\phi_{l-1}, \phi_{1})_{\tau} & \cdots & (\phi_{l-1}, \phi_{l-1})_{\tau}
	\end{bmatrix}
\end{equation}
$$



接下用矩阵来表示向量空间需要的变量，
$$
\bm\Phi = \begin{bmatrix}
\bm\phi & \bm0 \\
\bm0 & \bm\phi
\end{bmatrix}
$$

$$
\nabla \bm \Phi = 
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
$$

$$
\nabla \bm \Phi \cdot \bm \Phi = 
\begin{bmatrix}
\phi_0 \frac{\partial \phi_0}{\partial x} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial x} & 0 & \cdots & 0\\
0 & \cdots & 0 & \phi_{0} \frac{\partial \phi_{0}}{\partial y} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial y} 
\end{bmatrix}
$$

$$
(\nabla \bm \Phi \cdot \bm \Phi)^T = 
\begin{bmatrix}
 \phi_{0} \frac{\partial \phi_{0}}{\partial y} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial y} &0 & \cdots & 0 \\
 0 & \cdots & 0 & \phi_0 \frac{\partial \phi_0}{\partial x} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial x}
\end{bmatrix}
$$


$$
\varepsilon(\bm\Phi) = 
\begin{bmatrix}
	\begin{bmatrix}
		\frac{\partial \phi_0}{\partial x} & \frac{1}{2}\frac{\partial \phi_0}{\partial y}\\
		\frac{1}{2}\frac{\partial \phi_0}{\partial y} & 0
	\end{bmatrix}
	& \cdots
	& \begin{bmatrix}
		\frac{\partial \phi_{l_1}}{\partial x} & \frac{1}{2}\frac{\partial \phi_{l-1}}{\partial y}\\
		\frac{1}{2}\frac{\partial \phi_{l-1}}{\partial y} & 0
	\end{bmatrix}
	& \begin{bmatrix}
		0 & \frac{1}{2}\frac{\partial \phi_{0}}{\partial x} \\
		\frac{1}{2}\frac{\partial \phi_{0}}{\partial x} & \frac{\partial \phi_{0}}{\partial y}
	\end{bmatrix}
	& \cdots
	& \begin{bmatrix}
		0 & \frac{1}{2}\frac{\partial \phi_{l-1}}{\partial x} \\
		\frac{1}{2}\frac{\partial \phi_{l-1}}{\partial x} & \frac{\partial \phi_{l-1}}{\partial y}
	\end{bmatrix}
\end{bmatrix}
$$

$$
(\bm\Phi,\bm\Phi) = \int_{\tau} 
\begin{bmatrix}
\bm\phi^T & \bm0 \\
\bm0 & \bm\phi^T
\end{bmatrix} 

\begin{bmatrix}
\bm\phi & \bm0 \\
\bm0 & \bm\phi
\end{bmatrix}
\bm d \bm x = 
\begin{bmatrix}
		\bm H & 0 \\
		0 & \bm H
	\end{bmatrix}
$$

$$
(\nabla \bm \Phi \cdot \bm \Phi,\bm \Phi) =
\begin{bmatrix}
\bm G_{00} & \bm 0\\
\bm 0 & \bm G_{11}\\
\end{bmatrix}
$$

$$
\bm G_{00} = \int_\tau (\bm \phi^T \bm \phi) diag(\frac{\partial\bm\phi}{\partial x}) 
 \bm d\bm x
$$


$$
\bm G_{11} = \int_\tau (\bm \phi^T \bm \phi) diag(\frac{\partial\bm\phi}{\partial y})
 \bm d\bm x
$$
其中$\bm I$为单位矩阵
$$
(\varepsilon(\bm \Phi),\varepsilon(\bm \Phi)) = \int_{\tau} \varepsilon(\bm\Phi): \varepsilon(\bm\Phi) \bm d \bm x  =  
\begin{bmatrix}
\bm E_{0,0} & \bm E_{0, 1}\\
\bm E_{1,0} & \bm E_{1, 1}\\
\end{bmatrix}
$$

$$
\bm E_{0, 0} = \int_\tau \frac{\partial\bm\phi^T}{\partial x}\frac{\partial\bm\phi}{\partial x}
+\frac{1}{2}\frac{\partial\bm\phi^T}{\partial y}\frac{\partial\bm\phi}{\partial y}\mathrm
d\bm x
$$

$$ { }
\bm E_{1, 1} = \int_\tau \frac{\partial\bm\phi^T}{\partial y}\frac{\partial\bm\phi}{\partial y}
+\frac{1}{2}\frac{\partial\bm\phi^T}{\partial x}\frac{\partial\bm\phi}{\partial x}\mathrm
d\bm x
$$

$$
\bm E_{0, 1} = \int_\tau \frac{1}{2}\frac{\partial\bm\phi^T}{\partial y}\frac{\partial\bm\phi}{\partial x}\mathrm
d\bm x
$$

第一步计算$u^*$​
$$
\begin{align*}
	&\frac{\rho}{\Delta t} (\bm u^*,\bm v) + \mu (\epsilon(\bm u^*),\epsilon(\bm v)) 
	- \frac{\mu}{2} (\nabla(\bm u^*) \cdot \bm n , \bm v)_{\partial \Omega} \\
	&= \frac{\rho}{\Delta t}(\bm u^n,\bm v) - \rho(\bm u^n \cdot \nabla \bm u^n,\bm v) 
	-\mu(\epsilon(\bm u^n),\epsilon(\bm v)) \\ &+ (p^n \bm I,\epsilon(\bm v)) - (p^n \bm n ,\bm v)_{\partial \Omega}
	+ \frac{\mu}{2}(\nabla(\bm u^n) \cdot \bm n ,\bm v)_{\partial \Omega}
\end{align*}
$$
左边矩阵为
$$
\frac{\rho}{\Delta t}
\begin{bmatrix}
	\bm H & 0 \\
	0 & \bm H
\end{bmatrix}
+\mu
\begin{bmatrix}
\bm E_{0,0} & \bm E_{0, 1}\\
\bm E_{1,0} & \bm E_{1, 1}\\
\end{bmatrix}
-\frac{\mu}{2}
\begin{bmatrix}
\bm G_{00} & \bm 0\\
\bm 0 & \bm G_{11}\\
\end{bmatrix}_{\partial \Omega}
\begin{bmatrix}
\bm n_x & \bm n_y\\
\bm n_x & \bm n_y\\
\end{bmatrix}
$$

右边矩阵为
$$
-\rho\begin{bmatrix}
\bm G_{00} & \bm 0\\
\bm 0 & \bm G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\bm u^n_x \bm u^n_x\\
\bm u^n_y \bm u^n_y\\
\end{bmatrix}
-\rho\begin{bmatrix}
\bm G_{00} & \bm 0\\
\bm 0 & \bm G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\bm u^n_x \bm u^n_y\\
\bm u^n_x \bm u^n_y\\
\end{bmatrix} \\
+
(\frac{\rho}{\Delta t}
\begin{bmatrix}
	\bm H & 0 \\
	0 & \bm H
\end{bmatrix}
-\mu
\begin{bmatrix}
\bm E_{0,0} & \bm E_{0, 1}\\
\bm E_{1,0} & \bm E_{1, 1}\\
\end{bmatrix}
+\frac{\mu}{2}
\begin{bmatrix}
\bm G_{00} & \bm 0\\
\bm 0 & \bm G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\bm n_x & \bm n_y\\
\bm n_x & \bm n_y\\
\end{bmatrix}
)
\begin{bmatrix}
		u_{\bm x}\\
		u_{\bm y}
	\end{bmatrix}
$$


## 参考文献

1. [Fenics examples for the Navier-Stokes
   equations](https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1004.html#the-navier-stokes-equations)
