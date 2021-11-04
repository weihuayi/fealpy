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

若其为牛顿流体的话，流体的切应力与应变时间(速度梯度)成正比。其关系如下

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

# 2. 变分格式

对两边乘上向量测试函数 $\boldsymbol v \in V$ 并在积分区域 $\Omega$ 上做积分

$$
\begin{aligned}
	\int_{\Omega} \rho \frac{\partial \boldsymbol u}{\partial t} \cdot \boldsymbol v
    \mathrm dx
	+ \int_{\Omega} \rho (\boldsymbol u \cdot \nabla \boldsymbol u) \cdot \boldsymbol v \mathrm dx 
	= 
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	+\mu \int_{\Omega}(\nabla \cdot (\nabla \boldsymbol u + \nabla (\boldsymbol u)^T)) \cdot \boldsymbol v \mathrm dx
	+\int_{\Omega} \rho \boldsymbol f \cdot \boldsymbol v \mathrm dx
\end{aligned}
$$

对以下几项用散度定理来处理

$$
\begin{aligned}
	-\int_{\Omega} \nabla p \cdot \boldsymbol v \mathrm dx 
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) - \nabla \cdot (p\boldsymbol v) \mathrm dx\\
	&= \int_{\Omega} p (\nabla \cdot \boldsymbol v) dx - \int_{\partial \Omega} p\boldsymbol v \cdot \boldsymbol n \mathrm ds
\end{aligned}
$$

和

$$
\begin{aligned}
	\int_{\Omega} (\nabla \cdot \nabla \boldsymbol u) \cdot \boldsymbol v \quad \mathrm dx 
	&= \int_{\Omega} \nabla \cdot (\nabla \boldsymbol u \cdot \boldsymbol v) - \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx\\
	&= \int_{\partial \Omega} \nabla \boldsymbol u \cdot \boldsymbol v  \cdot \boldsymbol n \mathrm ds - \int_{\Omega} \nabla \boldsymbol v : \nabla \boldsymbol u \mathrm dx
\end{aligned}
$$

因此可得到其变分形式

$$
\begin{aligned}
	(\rho \frac{\partial \boldsymbol u}{\partial t},\boldsymbol v) + (\rho \boldsymbol u \cdot \nabla \boldsymbol u ,\boldsymbol v ) 
	- ( p ,\nabla \cdot \boldsymbol v) + (p\boldsymbol n ,\boldsymbol v)_{\partial \Omega} \\
	+( \sigma (\boldsymbol u  + (\boldsymbol u)^T) , \nabla \boldsymbol v) 
	-( \sigma (\boldsymbol u + (\boldsymbol u)^T) \cdot \boldsymbol n ,  \boldsymbol v))_{\partial \Omega}
	 =  (\rho \boldsymbol f,\boldsymbol v)
\end{aligned}
$$

由于任何一个矩阵都可以分解成一个对称矩阵和反对称矩阵的求和，即

$$
\nabla \boldsymbol v = \frac{\nabla \boldsymbol v + (\nabla \boldsymbol v)^T}{2} + \frac{\nabla \boldsymbol v - (\nabla\boldsymbol v)^T}{2}
$$

易证反对称矩阵和对称矩阵求内积会消失，所以变分形式可以变为

$$
\begin{aligned}
	(\rho \frac{\partial \boldsymbol u}{\partial t},\boldsymbol v) + (\rho \boldsymbol u \cdot \nabla \boldsymbol u ,\boldsymbol v ) 
	- ( p ,\nabla \cdot \boldsymbol v) + (p\boldsymbol n ,\boldsymbol v)_{\partial \Omega} \\
	+( \sigma(\boldsymbol u) , \varepsilon(\boldsymbol v)) 
	-( \sigma(\boldsymbol u) \cdot \boldsymbol n ,  \boldsymbol v))_{\partial \Omega}
	=  (\rho \boldsymbol f,\boldsymbol v)
\end{aligned}`1
$$


## 3.3 Oseen算法

非线性项 $\boldsymbol u \cdot \nabla \boldsymbol u $ 进行线性化处理 $ \boldsymbol u^n \cdot \nabla \boldsymbol u^{n+1}$

$$
\begin{aligned}
	( \frac{ \boldsymbol u^{n+1}-\boldsymbol u^{n}}{\Delta t},\boldsymbol v) + ( \boldsymbol u^{n} \cdot \nabla \boldsymbol u^{n+1} ,\boldsymbol v ) 
	- ( p^{n+1} ,\nabla \cdot \boldsymbol v) +  (\nabla  \boldsymbol u^{n+1}  , \nabla \boldsymbol v) 
	 &=  ( \boldsymbol f^{n+1},\boldsymbol v) \\
	 ( \nabla \cdot\boldsymbol u^{n+1}, q) &= 0
\end{aligned}
$$

$$
\left[\begin{array}{lll}
E+A+D & 0 &-B_{1} \\
0 & E+A+D&-B_{2}\\
B_{1}^{T} & B_{2}^{T} &0
\end{array}\right]\left[\begin{array}{l}
\boldsymbol{u}_{0} \\
\boldsymbol{u}_{1} \\
\boldsymbol{p}
\end{array}\right]
$$

其中

$$
\begin{aligned}
E &=\frac{1}{\Delta t}\int_{\tau} \boldsymbol \phi^{T} \boldsymbol \phi d \boldsymbol x \\
A &= \int_{\tau}  \frac{\partial \boldsymbol \phi^T}{\partial x} \frac{\partial \boldsymbol \phi}{\partial x} + \frac{\partial \boldsymbol \phi^T}{\partial y} \frac{\partial \boldsymbol \phi}{\partial y}d \boldsymbol x \\
D &=  \int_{\tau} \boldsymbol u_0^{n} \boldsymbol \phi^T  \frac{\partial \boldsymbol \phi}{\partial x} + \boldsymbol u_1^{n} \boldsymbol \phi^T  \frac{\partial \boldsymbol \phi}{\partial y}d \boldsymbol x \\
B_1 &=  \int_{\tau} \frac{\partial \boldsymbol \phi^T}{\partial x} \boldsymbol \varphi d \boldsymbol x \\
B_2 &=  \int_{\tau} \frac{\partial \boldsymbol \phi^T}{\partial y} \boldsymbol \varphi d \boldsymbol x 
\end{aligned}
$$

# 4 Benchmark

## 4.1 Channel flow(Poisuille flow)

$\qquad$Channel flow是对壁面固定管道内流动情况的模拟

另$\Omega = [0,1]\times[0,1],\rho =1,\mu = 1,\boldsymbol f = 0$，方程可以变为

$$
\begin{aligned}
\frac{\partial \boldsymbol u}{\partial t}+\boldsymbol u \cdot \nabla\boldsymbol u  &= -\nabla p +  \Delta \boldsymbol u \qquad in \quad \Omega\times(0,T) \\
\nabla \cdot \boldsymbol u &= 0 \qquad in \quad \Omega\times(0,T)\\
\boldsymbol u &= 0 \qquad on \quad \Omega\times\{0\} \\ 
\boldsymbol u &= 0 \qquad on \quad [0,1] \times \{0,1\} \times[0,T]   \\ 
p &= 8 \qquad  on \quad \{ 0 \} \times [0,1] \times[0,T] \\ 
p &= 0 \qquad  on \quad \{ 1 \} \times [0,1] \times[0,T] \\ 
\end{aligned}
$$

其解析解为$u = (4y(1-y),0),p = 8(1-x)$



# 5 基函数表示

给定 $\Omega$​ 上一个单纯形网格离散 $\tau$, 构造连续的分k次$Lgarange$多项式空间, 其基函数向量记为：

$$
\begin{aligned}
\boldsymbol\phi := [\phi_0(\boldsymbol x), \phi_1(\boldsymbol x), \cdots, \phi_{l-1}(\boldsymbol x)],
\forall \boldsymbol x \in \tau
\end{aligned}
$$


$$
\begin{aligned}
\nabla \boldsymbol \phi = 
\begin{bmatrix}
	\frac{\partial \phi_0}{\partial x} & \frac{\partial \phi_1}{\partial x} & \cdots \frac{\partial \phi_{l-1}}{\partial x} \\
	\frac{\partial \phi_0}{\partial y} & \frac{\partial \phi_1}{\partial y} & \cdots \frac{\partial \phi_{l-1}}{\partial y}
\end{bmatrix}
:=\begin{bmatrix}
	\frac{\partial \boldsymbol \phi}{\partial x} \\
	\frac{\partial \boldsymbol \phi}{\partial y}
\end{bmatrix}
\end{aligned}
$$

设网格边界边（2D)或边界面（3D)上的**局部基函数**个数为 $m$ 个，其组成的**行向量函数**记为
$$
\boldsymbol\omega (\boldsymbol x) = \left[\omega_0(\boldsymbol x), \omega_1(\boldsymbol x), \cdots, \omega_{m-1}(\boldsymbol x)\right]
$$


则k次向量空间$\mathcal P_k(K;\mathcal R^2)$的基函数为

$$
\begin{aligned}
\boldsymbol\Phi = \begin{bmatrix}
\boldsymbol\phi & \boldsymbol0 \\
\boldsymbol0 & \boldsymbol\phi
\end{bmatrix}
\end{aligned}
$$

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



$$
\begin{aligned}
\varepsilon(\boldsymbol\Phi) = 
\begin{bmatrix}
	\begin{bmatrix}
		\frac{\partial \phi_0}{\partial x} & \frac{1}{2}\frac{\partial \phi_0}{\partial y}\\
		\frac{1}{2}\frac{\partial \phi_0}{\partial y} & 0
	\end{bmatrix}
	& \cdots
	& \begin{bmatrix}
		\frac{\partial \phi_{l-1}}{\partial x} & \frac{1}{2}\frac{\partial \phi_{l-1}}{\partial y}\\
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
\end{aligned}
$$



## 5.1函数的表示

- $\boldsymbol u = (u_x,u_y)$

$$
\begin{aligned}
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

\end{aligned}
$$

- $\nabla \boldsymbol u$

$$
\begin{aligned}
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
\end{aligned}
$$

- $\boldsymbol u \cdot \nabla \boldsymbol u$

$$
\begin{aligned}
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
\end{aligned}
$$

- $\epsilon(\boldsymbol u) = \frac{1}{2}(\nabla \boldsymbol u + (\nabla \boldsymbol u)^T)$

$$
\begin{aligned}		
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
\end{aligned}
$$



## 5.2矩阵表示
$$
\begin{aligned}
    \boldsymbol H=\int_{\tau} \boldsymbol \phi^{T} \boldsymbol \phi d \boldsymbol x=
    \begin{bmatrix}
        (\phi_{0}, \phi_{0})_{\tau} & (\phi_{0}, \phi_{1})_{\tau} & \cdots & (\phi_{0}, \phi_{l-1})_{\tau} \\
        (\phi_{1}, \phi_{0})_{\tau} & (\phi_{1}, \phi_{1})_{\tau} & \cdots & (\phi_{1}, \phi_{l-1})_{\tau} \\
        \vdots & \vdots & \ddots & \vdots \\
        (\phi_{l-1}, \phi_{0})_{\tau} & (\phi_{l-1}, \phi_{1})_{\tau} & \cdots & (\phi_{l-1}, \phi_{l-1})_{\tau}
    \end{bmatrix}
\end{aligned}
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


$$
(p^n\boldsymbol n ,\boldsymbol v)_{\partial \tau} =
\begin{bmatrix}
G_0 \\
G_1
\end{bmatrix}
$$

$$
G_0 =
\int_{\partial \tau} 
\boldsymbol\omega^T (p^nn_0)
\boldsymbol d \boldsymbol x
$$

$$
G_1 =
\int_{\partial \tau} 
\boldsymbol\omega^T (p^nn_1)
\boldsymbol d \boldsymbol x
$$


$$
( \epsilon(\boldsymbol \Phi) \cdot \boldsymbol n ,  \boldsymbol \Phi))_{\partial \tau}
=
\int_{\partial \tau} 
\begin{bmatrix}
\boldsymbol\phi^T & \boldsymbol0 \\
\boldsymbol0 & \boldsymbol\phi^T
\end{bmatrix} 

\begin{bmatrix}
\frac{\partial \boldsymbol \phi}{\partial x}n_0+\frac{1}{2}\frac{\partial \boldsymbol \phi}{\partial y}n_1 & \frac{1}{2}\frac{\partial \boldsymbol \phi}{\partial x}n_1 \\
\frac{1}{2}\frac{\partial \boldsymbol \phi}{\partial y}n_0 &\frac{1}{2}\frac{\partial \boldsymbol\phi}{\partial x}n_0+\frac{\partial \boldsymbol \phi}{\partial y}n_1 
\end{bmatrix}
\boldsymbol d \boldsymbol x
\\=
\begin{bmatrix}
\boldsymbol D_{00} & \boldsymbol D_{01} \\
\boldsymbol D_{10} & \boldsymbol D_{11} \\
\end{bmatrix}
$$


$$
\boldsymbol D_{00} = \int_{\partial\tau} \boldsymbol\omega^T\frac{\partial\boldsymbol\phi}{\partial x}n_0
+\frac{1}{2}\boldsymbol\omega^T\frac{\partial\boldsymbol\phi^T}{\partial y}n_1\mathrm
d\boldsymbol x
$$


$$
\boldsymbol D_{11} = \int_{\partial\tau}  \boldsymbol\omega^T\frac{\partial\boldsymbol\phi}{\partial y}n_1
+\frac{1}{2}\boldsymbol\omega^T\frac{\partial\boldsymbol\phi^T}{\partial x}n_0\mathrm
d\boldsymbol x
$$

$$
\begin{aligned}
\boldsymbol D_{01} = 
\int_{\partial \tau} \frac{1}{2} \boldsymbol \omega^T \frac{\partial\boldsymbol\phi^T}{\partial x} n_1 \mathrm
d\boldsymbol x
\end{aligned}
$$


$$
\boldsymbol D_{10} = 
\int_{ \partial\tau }\frac{1}{2}\boldsymbol\omega^T\frac{\partial\boldsymbol\phi^T}{\partial y}n_0\mathrm
d\boldsymbol x
$$

$$

$$






## 参考文献

1. [Fenics examples for the Navier-Stokes
   
   equations](https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1004.html#the-navier-stokes-equations):
