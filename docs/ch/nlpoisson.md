# 非线性 Poisson 方程求解



首先给出一个扩散系数为非线性的例子

$$
-\nabla\left(a(u)\nabla u\right) = f
$$

满足如下的边界条件：

$$
u = g_D, \quad\text{on }\Gamma_D\leftarrow \text{\bf Dirichlet } 
$$

$$
\frac{\partial u}{\partial\boldsymbol n}  = g_N, \quad\text{on }\Gamma_N \leftarrow \text{\bf Neumann}
$$

$$
\frac{\partial u}{\partial\boldsymbol n} + \kappa u = g_R, \quad\text{on }\Gamma_R \leftarrow \text{\bf Robin}
$$

在 Poisson 方程两端分别乘以测试函数 $v \in H_{D,0}^1(\Omega)$, 利用分部积分，可得到其对应的**连续弱形式**

$$
(a(u)\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_R} = (f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

对连续的弱形式线性化

$$
u = u^0 + \delta u
$$


$$
(a(u^0+\delta u)\nabla (u^0+\delta u),\nabla v)+<\kappa u^0+\delta u, v>_{\Gamma_R} = (f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$


非线性扩散系数进行 Taylor 展开

$$
a(u^0 + \delta u) = a(u^0) + a_u'(u^0)\delta u + \mathcal O(\delta u^2)
$$

忽略掉二次项

$$
(a(u^0)\nabla\delta u, \nabla v) + (a_u'(u^0)\nabla u^0\cdot\delta u, \nabla v) 
+ <\kappa\delta u, v>_{\Gamma_R}
= 
(f,v) - (a(u^0)\nabal u^0, \nabla v) - <\kappa u^0, v>_{\Gamma_R} + <g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

设网格单元 $\tau$ 上的基函数向量为

$$
\bm\phi = \left[\phi_0, \phi_1, \cdots, \phi_{ldof-1}
$$










第二个是反应项为非线性的例子

$$
-\nabla\left(\nabla u\right) + u^3=f 
$$


满足如下的边界条件：

$$
u = g_D, \quad\text{on }\Gamma_D\leftarrow \text{\bf Dirichlet } 
$$

$$
\frac{\partial u}{\partial\boldsymbol n}  = g_N, \quad\text{on }\Gamma_N \leftarrow \text{\bf Neumann}
$$

$$
\frac{\partial u}{\partial\boldsymbol n} + \kappa u = g_R, \quad\text{on }\Gamma_R \leftarrow \text{\bf Robin}
$$
