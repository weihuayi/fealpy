---
title: Navier-stokes 方程数值求解
permalink: /docs/zh/num-navier-stokes-equation
key: docs-num-navier-stoke-equation-zh
author: wpx
---

## PDE 模型

不可压的流体

$$
\begin{cases}
\rho (\frac{\partial \boldsymbol u}{\partial t}+\boldsymbol u \cdot \nabla\boldsymbol u)  = 
-\nabla p + \nabla \cdot \sigma(\boldsymbol u) +\rho \boldsymbol f \\
\nabla \cdot \boldsymbol u = 0
\end{cases}
$$

若其为牛顿流体的话，流体的切应力与应变时间(速度梯度)成正比。其关系如下

$$
\sigma(\boldsymbol u) = 2 \mu \varepsilon(\boldsymbol u)
$$

$$
\varepsilon(\boldsymbol u) = \frac{1}{2} (\nabla \boldsymbol u + (\nabla \boldsymbol u)^T)
$$

其中符号的物理意义分别为

- $\boldsymbol u$ 代表速度
- $p$ 代表单位面积上的压力
- $f$ 单位质量流体微团的体积力
- $\mu$ 分子粘性系数

## 变分格式

对两便乘上向量测试函数 $\boldsymbol v \in V$ 并在积分区域 $\Omega$ 上做积分

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

## channel 

## ipcs算法

- 第一步计算中间速度$u^*$

$$
\rho (\frac{u^* - u^n}{\Delta t},v) + (u^n \cdot \nabla u^n,v)
$$



## 基函数表示

给定 $\Omega$​ 上一个单纯形网格离散 $\tau$, 构造连续的分p 次$Lgarange$多项式空间, 其基函数向量记为：

$$
\boldsymbol\phi := [\phi_0(\boldsymbol x), \phi_1(\boldsymbol x), \cdots, \phi_{l-1}(\boldsymbol x)],
\forall \boldsymbol x \in \tau
$$


$$
\nabla \boldsymbol \phi = 
\begin{bmatrix}
	\frac{\partial \phi_0}{\partial x} & \frac{\partial \phi_1}{\partial x} & \cdots \frac{\partial \phi_{l-1}}{\partial x} \\
	\frac{\partial \phi_0}{\partial y} & \frac{\partial \phi_1}{\partial y} & \cdots \frac{\partial \phi_{l-1}}{\partial y}
\end{bmatrix}
:=\begin{bmatrix}
	\frac{\partial \boldsymbol \phi}{\partial x} \\
	\frac{\partial \boldsymbol \phi}{\partial y}
\end{bmatrix}
$$



则k次向量空间$\mathcal P_k(K;\mathcal R^2)$的基函数为

$$
\boldsymbol\Phi = \begin{bmatrix}
\boldsymbol\phi & \boldsymbol0 \\
\boldsymbol0 & \boldsymbol\phi
\end{bmatrix}
$$

$$
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
$$

$$
\boldsymbol \Phi  \cdot \nabla \boldsymbol \Phi = 
\begin{bmatrix}
\phi_0 \frac{\partial \phi_0}{\partial x} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial x} & 0 & \cdots & 0\\
0 & \cdots & 0 & \phi_{0} \frac{\partial \phi_{0}}{\partial y} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial y} 
\end{bmatrix}
$$

$$
(\boldsymbol \Phi \cdot \nabla \boldsymbol \Phi )^T = 
\begin{bmatrix}
 \phi_{0} \frac{\partial \phi_{0}}{\partial y} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial y} &0 & \cdots & 0 \\
 0 & \cdots & 0 & \phi_0 \frac{\partial \phi_0}{\partial x} & \cdots & \phi_{l-1} \frac{\partial \phi_{l-1}}{\partial x}
\end{bmatrix}
$$


$$
\varepsilon(\boldsymbol\Phi) = 
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



## 函数的表示

- $\boldsymbol u = (u_x,u_y)$

$$
\boldsymbol u = (u_x,u_y) = (\phi_{i},0)u_{x_i}+(0,\phi_{i})u_{y_i} = 
\begin{bmatrix}
	\boldsymbol\phi & 0 \\
	0 & \boldsymbol\phi
\end{bmatrix}
\begin{bmatrix}
	u_{x_0} \\
	\vdots \\
	u_{x_{N-1}}\\
	u_{y_0} \\
	\vdots \\
	u_{y_{N-1}}\\
\end{bmatrix}
:=
\begin{bmatrix}
	\boldsymbol\phi & 0 \\
	0 & \boldsymbol\phi
\end{bmatrix}
\begin{bmatrix}
	\boldsymbol u_{x}\\
	\boldsymbol u_{y}
\end{bmatrix}
$$

- $\nabla \boldsymbol u$

$$
\begin{align*}
	\nabla \boldsymbol u &=  
			\begin{bmatrix}
				 \frac{\partial u_x}{\partial x} & \frac{\partial u_y}{\partial x} \\
				 \frac{\partial u_x}{\partial y} & \frac{\partial u_y}{\partial y}
				\end{bmatrix} 
			=
			\begin{bmatrix}
				 u_{x_i}\frac{\partial \phi_i}{\partial x} & u_{y_i}\frac{\partial \phi_i}{\partial x}  \\
				 u_{x_i}\frac{\partial \phi_i}{\partial y} &  u_{y_i}\frac{\partial \phi_i}{\partial y}
			\end{bmatrix}\\
			&=
			\begin{bmatrix}
				\frac{\partial \phi_i}{\partial x} & 0\\
				\frac{\partial \phi_i}{\partial y} & 0
			\end{bmatrix}
			u_{x_i}
		    +
		    \begin{bmatrix}
		    	0 & \frac{\partial \phi_i}{\partial x} \\
		    	0 & \frac{\partial \phi_i}{\partial y} 
		    \end{bmatrix}
	    	u_{y_i}
			=\nabla \Phi 
			\begin{bmatrix}
				\boldsymbol u_{x}\\
				\boldsymbol u_{y}
			\end{bmatrix}\\	
\end{align*}
$$

- $\boldsymbol u \cdot \nabla \boldsymbol u$

$$
\begin{align*}
	\boldsymbol u \cdot \nabla \boldsymbol u &=  
	[u_x,u_y]
	\begin{bmatrix}
		\frac{\partial u_x}{\partial x} & \frac{\partial u_y}{\partial x} \\
		\frac{\partial u_x}{\partial y} & \frac{\partial u_y}{\partial y}
	\end{bmatrix} 
	\\&=
	\begin{bmatrix}
	u_{x_i}u_{x_i} \phi_i \frac{\partial \phi_i}{\partial x} +
	u_{x_i}u_{y_i} \phi_i \frac{\partial \phi_i}{\partial y} , 
	u_{x_i}u_{y_i} \phi_i \frac{\partial \phi_i}{\partial x} +
	u_{y_i}u_{y_i} \phi_i \frac{\partial \phi_i}{\partial y}
	\end{bmatrix}
	\\&=
	\begin{bmatrix}
		\phi_i\frac{\partial \phi_i}{\partial x},
		0
	\end{bmatrix}
	u_{x_i}u_{x_i}
	+
	\begin{bmatrix}
		0  ,
		\phi_i\frac{\partial \phi_i}{\partial y}
	\end{bmatrix}
	 u_{y_i}u_{y_i}
	+
	\begin{bmatrix}
		\phi_i\frac{\partial \phi_i}{\partial y},
		0
	\end{bmatrix}
	u_{x_i}u_{y_i}
	+ 
	\begin{bmatrix}
		0  ,
		\phi_i\frac{\partial \phi_i}{\partial x}
	\end{bmatrix}
	u_{y_i}u_{x_i}
	\\& =\boldsymbol \Phi \cdot \nabla \boldsymbol \Phi 
	\begin{bmatrix}
		\boldsymbol u_{x}\boldsymbol u_{x}\\
		\boldsymbol u_{y}\boldsymbol u_{y}
	\end{bmatrix}
	+ (\boldsymbol \Phi \cdot \nabla \boldsymbol \Phi)^T
	\begin{bmatrix}
		\boldsymbol u_{x}\boldsymbol u_{y}\\
		\boldsymbol u_{x}\boldsymbol u_{y}
	\end{bmatrix}
\end{align*}
$$

- $\epsilon(\boldsymbol u) = \frac{1}{2}(\nabla \boldsymbol u + (\nabla \boldsymbol u)^T) $

$$
\begin{align*}		
	\epsilon(\boldsymbol u) &= \frac{1}{2} (\nabla \boldsymbol u + (\nabla \boldsymbol u)^T) \\
			 &= \frac{1}{2}
			 \begin{bmatrix}
				2 u_{x_i}\frac{\partial \phi_i}{\partial x} & u_{y_i}\frac{\partial \phi_i}{\partial x} + u_{x_i}\frac{\partial \phi_i}{\partial y} \\
				u_{y_i}\frac{\partial \phi_i}{\partial x} + u_{x_i}\frac{\partial \phi_i}{\partial y} & 2 u_{y_i}\frac{\partial \phi_i}{\partial y}
			\end{bmatrix} \\
			&=
			\begin{bmatrix}
				\frac{\partial \phi_i}{\partial x} & \frac{1}{2}\frac{\partial \phi_i}{\partial y}\\
				\frac{1}{2} \frac{\partial \phi_i}{\partial y} & 0
			\end{bmatrix}
			u_{x_i}
			+
			\begin{bmatrix}
				0 & \frac{1}{2}\frac{\partial \phi_i}{\partial x}\\
				\frac{1}{2} \frac{\partial \phi_i}{\partial x} & \frac{\partial \phi_i}{\partial y}
			\end{bmatrix}
			u_{y_i}
			=\epsilon (\Phi)
			\begin{bmatrix}
				\boldsymbol u_{x}\\
				\boldsymbol u_{y}
			\end{bmatrix}
\end{align*}
$$



## 矩阵表示

$$
\begin{equation}
    \boldsymbol H=\int_{\tau} \boldsymbol \phi^{T} \boldsymbol \phi d \boldsymbol x=
    \begin{bmatrix}
        (\phi_{0}, \phi_{0})_{\tau} & (\phi_{0}, \phi_{1})_{\tau} & \cdots & (\phi_{0}, \phi_{l-1})_{\tau} \\
        (\phi_{1}, \phi_{0})_{\tau} & (\phi_{1}, \phi_{1})_{\tau} & \cdots & (\phi_{1}, \phi_{l-1})_{\tau} \\
        \vdots & \vdots & \ddots & \vdots \\
        (\phi_{l-1}, \phi_{0})_{\tau} & (\phi_{l-1}, \phi_{1})_{\tau} & \cdots & (\phi_{l-1}, \phi_{l-1})_{\tau}
    \end{bmatrix}
\end{equation}
$$


$$
(\boldsymbol\Phi,\boldsymbol\Phi) = \int_{\tau} 
\begin{bmatrix}
\boldsymbol\phi^T & \boldsymbol0 \\
\boldsymbol0 & \boldsymbol\phi^T
\end{bmatrix} 

\begin{bmatrix}
\boldsymbol\phi & \boldsymbol0 \\
\boldsymbol0 & \boldsymbol\phi
\end{bmatrix}
\boldsymbol d \boldsymbol x = 
\begin{bmatrix}
		\boldsymbol H & 0 \\
		0 & \boldsymbol H
	\end{bmatrix}
$$

$$
(\nabla \boldsymbol \Phi \cdot \boldsymbol \Phi,\boldsymbol \Phi) =
\begin{bmatrix}
\boldsymbol G_{00} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol G_{11}\\
\end{bmatrix}
$$

$$
\boldsymbol G_{00} = \int_\tau (\boldsymbol \phi^T \boldsymbol \phi) diag(\frac{\partial\boldsymbol\phi}{\partial x}) 
 \boldsymbol d\boldsymbol x
$$

$$
\boldsymbol G_{11} = \int_\tau (\boldsymbol \phi^T \boldsymbol \phi) diag(\frac{\partial\boldsymbol\phi}{\partial y})
 \boldsymbol d\boldsymbol x
$$

$$
(\varepsilon(\boldsymbol \Phi),\varepsilon(\boldsymbol \Phi)) = \int_{\tau} \varepsilon(\boldsymbol\Phi): \varepsilon(\boldsymbol\Phi) \boldsymbol d \boldsymbol x  =  
\begin{bmatrix}
\boldsymbol E_{0,0} & \boldsymbol E_{0, 1}\\
\boldsymbol E_{1,0} & \boldsymbol E_{1, 1}\\
\end{bmatrix}
$$

$$
\boldsymbol E_{0, 0} = \int_\tau \frac{\partial\boldsymbol\phi^T}{\partial x}\frac{\partial\boldsymbol\phi}{\partial x}
+\frac{1}{2}\frac{\partial\boldsymbol\phi^T}{\partial y}\frac{\partial\boldsymbol\phi}{\partial y}\mathrm
d\boldsymbol x
$$

$$
\boldsymbol E_{1, 1} = \int_\tau \frac{\partial\boldsymbol\phi^T}{\partial y}\frac{\partial\boldsymbol\phi}{\partial y}
+\frac{1}{2}\frac{\partial\boldsymbol\phi^T}{\partial x}\frac{\partial\boldsymbol\phi}{\partial x}\mathrm
d\boldsymbol x
$$

$$
\boldsymbol E_{0, 1} = \int_\tau \frac{1}{2}\frac{\partial\boldsymbol\phi^T}{\partial y}\frac{\partial\boldsymbol\phi}{\partial x}\mathrm
d\boldsymbol x
$$

## ipcs算法

第一步计算$u^{*}$
$$
\begin{align*}
	&\frac{\rho}{\Delta t} (\boldsymbol u^*,\boldsymbol v) + \mu (\epsilon(\boldsymbol u^*),\epsilon(\boldsymbol v)) 
	- \frac{\mu}{2} (\nabla(\boldsymbol u^*) \cdot \boldsymbol n , \boldsymbol v)_{\partial \Omega} \\
	&= \frac{\rho}{\Delta t}(\boldsymbol u^n,\boldsymbol v) - \rho(\boldsymbol u^n \cdot \nabla \boldsymbol u^n,\boldsymbol v) 
	-\mu(\epsilon(\boldsymbol u^n),\epsilon(\boldsymbol v)) \\ &+ (p^n \boldsymbol I,\epsilon(\boldsymbol v)) - (p^n \boldsymbol n ,\boldsymbol v)_{\partial \Omega}
	+ \frac{\mu}{2}(\nabla(\boldsymbol u^n) \cdot \boldsymbol n ,\boldsymbol v)_{\partial \Omega}的
\end{align*}
$$

左边矩阵为

$$
\frac{\rho}{\Delta t}
\begin{bmatrix}
	\boldsymbol H & 0 \\
	0 & \boldsymbol H
\end{bmatrix}
+\mu
\begin{bmatrix}
\boldsymbol E_{0,0} & \boldsymbol E_{0, 1}\\
\boldsymbol E_{1,0} & \boldsymbol E_{1, 1}\\
\end{bmatrix}
-\frac{\mu}{2}
\begin{bmatrix}
\boldsymbol G_{00} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol G_{11}\\
\end{bmatrix}_{\partial \Omega}
\begin{bmatrix}
\boldsymbol n_x & \boldsymbol n_y\\
\boldsymbol n_x & \boldsymbol n_y\\
\end{bmatrix}
$$

右边矩阵为

$$
-\rho\begin{bmatrix}
\boldsymbol G_{00} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\boldsymbol u^n_x \boldsymbol u^n_x\\
\boldsymbol u^n_y \boldsymbol u^n_y\\
\end{bmatrix}
-\rho\begin{bmatrix}
\boldsymbol G_{00} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\boldsymbol u^n_x \boldsymbol u^n_y\\
\boldsymbol u^n_x \boldsymbol u^n_y\\
\end{bmatrix} \\
+
(\frac{\rho}{\Delta t}
\begin{bmatrix}
	\boldsymbol H & 0 \\
	0 & \boldsymbol H
\end{bmatrix}
-\mu
\begin{bmatrix}
\boldsymbol E_{0,0} & \boldsymbol E_{0, 1}\\
\boldsymbol E_{1,0} & \boldsymbol E_{1, 1}\\
\end{bmatrix}
+\frac{\mu}{2}
\begin{bmatrix}
\boldsymbol G_{00} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol G_{11}\\
\end{bmatrix}
\begin{bmatrix}
\boldsymbol n_x & \boldsymbol n_y\\
\boldsymbol n_x & \boldsymbol n_y\\
\end{bmatrix}
)
\begin{bmatrix}
		u_{\boldsymbol x}\\
		u_{\boldsymbol y}
	\end{bmatrix}
$$


## 参考文献

1. [Fenics examples for the Navier-Stokes
   equations](https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1004.html#the-navier-stokes-equations)
