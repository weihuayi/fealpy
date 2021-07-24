<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Navier-stokes 问题

## PDE 模型
不可压的流体
$$
\begin{equation}
\begin{cases}
\rho (\frac{\partial \bfu}{\partial t}+\bfu \cdot \nabla\bfu)  =　-\nabla p + \nabla \cdot \sigma(\bfu) +\rho \bff \\
\nabla \cdot \bfu = 0
\end{cases}
\end{equation}
$$

若其为牛顿流体的话，流体的切应力与应变时间(速度梯度)成正比。其关系如下

$$
\begin{align}
	\sigma(\bfu) &= 2 \mu \epsilon(\bfu) \\
	\epsilon(\bfu) &= \frac{1}{2} (\nabla \bfu + (\nabla \bfu)^T)
\end{align}
$$

其中符号的物理意义分别为

- $$\bfu=\bfu(u,v,w,t) $$ 代表速度
- p代表单位面积上的压力
- f单位质量流体微团的体积力
- $\mu$ 分子粘性系数

##　变分格式
对两便乘上向量测试函数$\bfv \in V$并在积分区域$\Omega$上做积分
$$
\begin{eqnarray}
	\int_{\Omega} \rho \frac{\partial \bfu}{\partial t}v dx
	+ \int_{\Omega} \rho \bfu \cdot \nabla \bfu \cdot \bfv dx 
	= 
	-\int_{\Omega} \nabla p \cdot \bfv dx 
	+\mu \int_{\Omega}(\nabla \cdot (\nabla \bfu + \nabla (\bfu)^T)) \cdot \bfv dx
	+\int_{\Omega} \rho \bff \cdot \bfv dx
\end{eqnarray}
$$

对以下几项用散度定理来处理

$$
\begin{align}
	-\int_{\Omega} \nabla p \cdot \bfv dx 
	&= \int_{\Omega} p (\nabla \cdot \bfv) - \nabla \cdot (p\bfv)dx\\
	&= \int_{\Omega} p (\nabla \cdot \bfv) dx - \int_{\partial \Omega} p\bfv \cdot \bfn ds
\end{align}
$$
和
$$
\begin{align}
	\int_{\Omega} (\nabla \cdot \nabla \bfu) \cdot \bfv \quad dx 
	&= \int_{\Omega} \nabla \cdot (\nabla \bfu \cdot \bfv) - \nabla \bfv : \nabla \bfu dx\\
	&= \int_{\partial \Omega} \nabla \bfu \cdot \bfv  \cdot \bfn ds - \int_{\Omega} \nabla \bfv : \nabla \bfu dx
\end{align}
$$

因此我们可得到其变分形式
$$
\begin{align}
	(\rho \frac{\partial \bfu}{\partial t},\bfv) + (\rho \bfu \cdot \nabla \bfu ,\bfv ) 
	- ( p ,\nabla \cdot \bfv) + (p\bfn ,\bfv)_{\partial \Omega} \\
	+(\mu \nabla (\bfu  + (\bfu)^T) , \nabla \bfv) 
	-(\mu \nabla (\bfu + (\bfu)^T) \cdot \bfn ,  \bfv))_{\partial \Omega}
	 =  (\rho \bff,\bfv)
\end{align}
$$

由于任何一个矩阵都可以分解成一个对称矩阵和反对称矩阵的求和，即
$$
\nabla v = \frac{\nabla v + (\nabla v)^T}{2} + \frac{\nabla v - (\nabla v)^T}{2}
$$

易证反对陈矩阵和对称矩阵求内积会消失，所以变分形式可以变为
$$
\begin{align}
	(\rho \frac{\partial \bfu}{\partial t},\bfv) + (\rho \bfu \cdot \nabla \bfu ,\bfv ) 
	- ( p ,\nabla \cdot \bfv) + (p\bfn ,\bfv)_{\partial \Omega} \\
	+( \nabla \sigma(\bfu) , \epsilon(\bfv)) 
	-( \nabla \sigma(\bfu) \cdot \bfn ,  \bfv))_{\partial \Omega}
	=  (\rho \bff,\bfv)
\end{align}
$$
