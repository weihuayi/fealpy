
# 有限元算法与程序实现

## 预备知识

给定向量函数 $\mathbf F(x)$, 其定义域为 $\Omega\in\mathbb R^n$, $\mathbf n$ 是 $\Omega$ 边界 $\partial \Omega$ 上的单位外法线向量.

$$
\int_{\Omega} \nabla\cdot\mathbf F~ \mathrm d x = \int_{\partial \Omega} F\cdot\mathbf n ~\mathrm d s
$$

这就是**散度定理**. 

$$
\begin{aligned}
\int_{\Omega} \nabla\cdot(v\nabla u)~\mathrm d x &= \int_{\partial\Omega} v\nabla u\cdot\mathbf n~\mathrm d s\\
\int_{\Omega} v\Delta u~\mathrm d x + \int_{\Omega}\nabla u\cdot\nabla v~\mathrm d x &= \int_{\partial\Omega} v\nabla u\cdot\mathbf n~\mathrm d s
\end{aligned}
$$

## 问题模型

考虑如下D氏边界条件的Poisson方程:

\begin{eqnarray}
-\Delta u(x) &=& f(x),\text{ on } \Omega.\label{eq:P}\\
u|_{\partial\Omega} &=& 0.
\end{eqnarray}

## Galerkin方法

有限元方法是一种基于 PDE (partial differential equations) 的变分形式 (variational formulation) 求解PDE近似解的方法.

引入函数空间 $H^1(\Omega)$, 对于任意 $v \in H^1(\Omega)$, $v$和它的一阶导数都在 $\Omega$ 上 $L^2$ 可积. 这里的 $H^1(\Omega)$ 是一个无限维的空间.

另外, 引入空间 $H^1_0(\Omega) := \{v\in H^1(\Omega), v|_{\partial\Omega} = 0\}$.

对于任意的 $v\in H^1_0(\Omega)$, 同乘以方程 \eqref{eq:P} 的两端, 然后做分部积分可得: 

\begin{equation}\label{eq:wg}
\int_{\Omega}\nabla u\cdot\nabla v\mathrm{d}x = \int_{\Omega}fv\mathrm{d}x,\quad\forall v \in H^1_0(\Omega).
\end{equation}

原问题就转化为: 求解 $u\in H_0^1(\Omega)$, 满足

\begin{equation}\label{eq:W}
a(u,v) = <f,v> \text{ for all }v\in H_0^1(\Omega).
\end{equation}

其中

$$
a(u,v) = \int_{\Omega}\nabla u\cdot\nabla v\mathrm{d}x,\quad <f,v> =  \int_{\Omega}fv\mathrm{d}x.
$$

下面我们考虑所谓 Galerkin 方法来求 \eqref{eq:W} 的逼近解. 上面 $H_0^1(\Omega)$ 是一个无限维的空间,
为了把无限维的问题转化为有限维的问题, 引入 $H_0^1(\Omega)$ 的一个有限维的子空间 $V$, 比如
$V=\mathrm{span}\{\phi_1,\phi_2,\ldots,\phi_N\}$. 对任何 $v \in V$, 它都有唯一的表示

$$
v = \sum\limits_{i=1}^N v_i\phi_i.
$$

可以看出空间 $V$ 和 $N$ 维向量空间 $\mathbb{R}^N$ 是同构的, 即

$$
v = \sum\limits_{i=1}^N v_i\phi_i\leftrightarrow\mathbf{v} =
\begin{pmatrix}
v_1 \\ v_2 \\ \vdots \\ v_N
\end{pmatrix}
$$

其中列向量 $\mathbf{v}$ 是 $v$ 在基 $\{\phi_i\}_{i=1}^N$ 的坐标. 

下面可以构造一个离散的问题: 求 $ \tilde u = \sum_{i=1}^{N}u_i \phi_i \in V$, 其对应的向量为 $\mathbf u$, 满足

\begin{equation}\label{eq:d}
a(\tilde u, v) = <f, v>,\quad\forall~v\in V.
\end{equation}

方程 \eqref{eq:d} 中仍然包含有无穷多个方程. 但 $V$ 是一个有限维空间, 本质上 $\tilde u= \sum_{i=1}^{N}u_i \phi_i$ 只需要满足下面 $N$ 方程即可

$$
\begin{cases}
a(\tilde u, \phi_1) = <f, \phi_1> \\
a(\tilde u, \phi_2) = <f, \phi_2> \\
\vdots \\
a(\tilde u, \phi_N) = <f, \phi_N> 
\end{cases}
$$

即
\begin{cases}
a(\sum_{i=1}^{N}u_i \phi_i, \phi_1) = <f, \phi_1> \\
a(\sum_{i=1}^{N}u_i \phi_i, \phi_2) = <f, \phi_2> \\
\vdots \\
a(\sum_{i=1}^{N}u_i \phi_i, \phi_N) = <f, \phi_N> 
\end{cases}

上面的 $N$ 方程可以改写为下面的形式:

$$
\begin{pmatrix}
a(\phi_1, \phi_1) & a(\phi_2, \phi_1) & \cdots & a(\phi_N, \phi_1) \\
a(\phi_1, \phi_2) & a(\phi_2, \phi_2) & \cdots & a(\phi_N, \phi_2) \\
\vdots & \vdots & \ddots & \vdots \\
a(\phi_1, \phi_N) & a(\phi_2, \phi_N) & \cdots & a(\phi_N, \phi_N) \\
\end{pmatrix}
\begin{pmatrix}
u_1 \\ u_2 \\ \vdots \\ u_N
\end{pmatrix}
= 
\begin{pmatrix}
<f, \phi_1> \\ <f, \phi_2> \\ \vdots \\ <f, \phi_N> 
\end{pmatrix}
$$

引入**刚度矩阵**(stiff matrix).
$$
\mathbf{A}=(a_{ij})_{N\times N}  
$$
其中 $a_{ij}=a(\phi_i,\phi_j)$.

和**载荷矢量**(load vector) 
$$
\mathbf{f} = \begin{pmatrix}
f_1\\ f_2 \\ \ldots \\f_N
\end{pmatrix} 
$$ 

其中 $ f_i=<f,\phi_i>$. 

可得到如下 $N$ 阶线性方程组:

$$
\mathbf{Au} = \mathbf{f}.
$$

求解可得原问题的逼近解:

$$
\tilde u = \sum\limits_{i=1}^N u_i\phi_i.
$$

## 有限元方法 

### 符号说明

| 符号 | 意义|
|:------:| :----|
| $\Omega$ | 求解区域 |
| $\mathcal{T}$ | $\Omega$ 上的三角形网格 |
| $N$ | $\mathcal{T}$ 的节点个数 |
| $NT$ | $\mathcal{T}$ 的单元个数 | 
| $x_i\in\mathbb{R}^2,i=1,\ldots,N$ | 网格节点 |
| $\tau := (x_i,x_j,x_k)$ | $\mathcal T$ 中由顶点 $(x_i,x_j,x_k)$ 构成三角形单元, 其中顶点按逆时针排序 |
| $e_{ij} := \vec{\mathbf x_i\mathbf x_j}$ | $\mathcal T$ 中以 $\mathbf x_i$ 和 $\mathbf x_j$ 为端点的一条边 |
| $\tau_{ij}$ | 表示边 $e_{ij}$ 从 $\mathbf x_i$ 看向 $\mathbf x_j$ 左手边的单元 |
| $(l_i, l_j, l_k)$ | 顶点 $(x_i,x_j,x_k)$ 对应的三边边长 |
| $W=\begin{pmatrix}0&-1\\1 & 0 \end{pmatrix}$ | 旋转矩阵, 作用在二维列向量上表示逆时针旋转该向量$90^\circ$ |
| $I=\begin{pmatrix} 1 & 0\\ 0& 1 \end{pmatrix}$ | 单位矩阵 |
| $|\tau_m|$ |  $\tau_m$ 的面积 |
| $\omega_i$ | $\mathcal T$ 中所有以 $x_i$ 为顶点的三角形单元集合 |
| $\xi_{ij}$ | 边 $e_{ij}$ 相邻三角形单元的集合, 如果有两个相邻三角形单元 $\tau_{ij}$ 和 $\tau_{ji}$, 则为**内部边**, 只有一个相邻单元 $\tau_{ij}$, 则为**边界边** |

### 重心坐标

给定三角形单元 $\tau$, 其三个顶点 $\mathbf{x}_i :=(x_i,y_i)$, $\mathbf{x}_j :=(x_j,y_j)$ 和 $\mathbf{x}_k :=(x_k,y_k)$ 逆时针排列, 且不在同一条直线上, 那么向量 $\vec{\mathbf{x}_i\mathbf{x}_j}$ 和 $\vec{\mathbf{x}_i\mathbf{x}_k}$ 是线性无关的. 这等价于矩阵

$$
A = 
\begin{pmatrix}
x_i & x_j & x_k \\
y_i & y_j & y_k \\
1   & 1   & 1 
\end{pmatrix}
$$

非奇异. 

任给一点 $\mathbf{x}:=(x,y)\in\tau_m$, 求解下面的线性方程组

$$
A 
\begin{pmatrix}
\lambda_i \\
\lambda_j\\
\lambda_k  
\end{pmatrix}
=\begin{pmatrix}
x \\
y\\
1  
\end{pmatrix}
$$

可得唯一的一组解$\lambda_i,\lambda_j,\lambda_k$. 

因此对任意二维点 $\mathbf{x}\in\tau$, 有

$$
\mathbf{x}=\lambda_i(\mathbf{x})\mathbf{x}_i + \lambda_j(\mathbf{x})\mathbf{x}_j + \lambda_k(\mathbf{x})\mathbf{x}_k 
\text{ with } \lambda_i(\mathbf{x}) + \lambda_j(\mathbf{x}) + \lambda_k(\mathbf{x}) = 1. 
$$

$\lambda_1,\lambda_2,\lambda_3$ 称为点 $\mathbf{x}$ 关于点 $\mathbf{x}_1,\mathbf{x}_2$ 和$\mathbf{x}_3$ 的**重心坐标**. 

重心坐标有它相应的几何意义. 给定 $\mathbf x\in\tau$, 把 $\tau$ 的第 $i$ 个顶点 $\mathbf{x}_i$ 换 $\mathbf{x}$
得到的三角形记为 $\tau_i(\mathbf{x})$, 则由克莱姆法则可得

\begin{equation}\label{eq:bc}
\lambda_i = {|\tau_i(\mathbf{x})| \over |\tau|}.
\end{equation}

其中 $|\cdot|$ 表示三角形的面积.

易知, $\lambda_1, \lambda_2, \lambda_3$ 都是关于 $\mathbf x$ 的线性函数, 且有

\begin{eqnarray*}
\lambda_1(\mathbf x_1) = 1,& \lambda_1(\mathbf x_2) = 0,& \lambda_1(\mathbf x_3) = 0\\
\lambda_2(\mathbf x_1) = 0,& \lambda_2(\mathbf x_2) = 1,& \lambda_2(\mathbf x_3) = 0\\
\lambda_3(\mathbf x_1) = 0,& \lambda_3(\mathbf x_2) = 0,& \lambda_3(\mathbf x_3) = 1\\
\end{eqnarray*}

$\lambda_1, \lambda_2, \lambda_3$ 关于 $\mathbf x$ 的梯度为:

$$
\begin{aligned}
\nabla\lambda_i = \frac{1}{2|\tau|}W\vec{\mathbf x_j\mathbf x_k}\\
\nabla\lambda_j = \frac{1}{2|\tau|}W\vec{\mathbf x_k\mathbf x_i}\\
\nabla\lambda_k = \frac{1}{2|\tau|}W\vec{\mathbf x_i\mathbf x_j}\\
\end{aligned}
$$

### 线性有限元基函数与空间

给定求解区域 $\Omega$ 上的一个三角形网格 $\mathcal T$, 它有 $N$ 个网格节点 $\{\mathbf x_i\}_{i=1}^N$, $NT$ 个三角形单元 $\{\tau_m\}_{m=1}^{NT}$.

![三角形网格](./figures/triangulation.png)

给定网格节点 $\mathbf x_i$, 记 $\omega_i$ 为 $\mathcal T$ 中所有以 $\mathbf x_i$ 为顶点的三角形单元集合, 即

$$
\omega_i = \{\tau,\,\mathbf x_i \text{ is one vertex of }\tau \in \mathcal T\}
$$

对每个网格节点 $\mathbf x_i$, 可以定义函数 
$$
\phi_i(\mathbf x) =
\begin{cases}
\lambda_i(\mathbf x),& \mathbf x \in \tau_m \in \omega_i\\
0, & \mathbf x \in \tau_m \notin \omega_i
\end{cases}
$$

由 $\phi_i(\mathbf x)$ 的定义和重心坐标函数的性质可知:
1. $\phi_i(\mathbf x_i)=1$,
1. $\phi_i(\mathbf x)$ 限止在 $\omega_i$ 中的每个单元 $\tau$ 上, 为 $\mathbf x_i$ 对应的重心坐标函数.
1. $\phi_i(\mathbf x)$ 在 $\omega_i$ 以外的单元上函数值为 0.
1. $\text{supp}(\phi_i)=\omega_i$.

因此, 我们可以说 $\phi_i$ 定义在 $\mathcal T$ 上的**分片线性连续函数**.

把 $\mathcal T$ 中的每个节点函数一起可以做为一组基, 张成一个**分片线性连续函数空间** 
$$
V = \text{span}\{\phi_1, \phi_2, \cdots, \phi_N\}
$$

### 刚度矩阵与右端载荷的计算

刚度矩阵 $A$ 的每一项
$$
a_{ij} = a(\phi_i, \phi_j) = \int_{\Omega}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x
$$

由 $\phi_i$ 的定义可知, 我们并不需要在整个 $\Omega$ 或者说整个 $\mathcal T$ 上来计算上面的积分, 只需要在 $\phi_i$ 和 $\phi_j$ 的支集的交集上计算, 即

$$
\begin{aligned}
&\int_{\Omega}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x\\
= &\int_{\omega_i\cap\omega_j}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x\\
= & 
\begin{cases}
\sum_{\tau\in\omega_i}\int_{\tau}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x, & i=j \\
\int_{\tau_{ij}}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x + \int_{\tau_{ji}}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x, & e_{ij}\text{ 为内部边} \\
\int_{\tau_{ij}}\nabla \phi_i\cdot\nabla \phi_j\mathrm d\mathbf x, & e_{ij}\text{ 为边界边} \\
0, & \mathbf x_i \text{ 与 } \mathbf x_j\text{ 不相邻}
\end{cases}\\
= & 
\begin{cases}
\sum_{\tau\in\omega_i}\int_{\tau}\nabla \lambda_i\cdot\nabla \lambda_j\mathrm d\mathbf x, & i=j \\
\int_{\tau_{ij}}\nabla \lambda_i\cdot\nabla \lambda_j\mathrm d\mathbf x + \int_{\tau_{ji}}\nabla \lambda_i\cdot\nabla \lambda_j\mathrm d\mathbf x, & e_{ij}\text{ 为内部边} \\
\int_{\tau_{ij}}\nabla \lambda_i\cdot\nabla \lambda_j\mathrm d\mathbf x, & e_{ij}\text{ 为边界边} \\
0, & \mathbf x_i \text{ 与 } \mathbf x_j\text{ 不相邻}
\end{cases}\\
\end{aligned}
$$

由以上推导可知, 我们实际上只需要在每个单元 $\tau$ 上计算出下面六个积分, 即可组装出刚度矩阵:
$$
\begin{aligned}
\int_{\tau}\nabla\lambda_i\cdot\nabla\lambda_i\,\mathrm d \mathbf x\\
\int_{\tau}\nabla\lambda_j\cdot\nabla\lambda_j\,\mathrm d \mathbf x\\
\int_{\tau}\nabla\lambda_k\cdot\nabla\lambda_k\,\mathrm d \mathbf x\\
\int_{\tau}\nabla\lambda_i\cdot\nabla\lambda_j\,\mathrm d \mathbf x\\
\int_{\tau}\nabla\lambda_i\cdot\nabla\lambda_k\,\mathrm d \mathbf x\\
\int_{\tau}\nabla\lambda_k\cdot\nabla\lambda_j\,\mathrm d \mathbf x\\
\end{aligned}
$$
上面的积分是可以直接算出来.

类似上面分解的思想, 右端载荷向量中每一个分量, 可做如下分解:
$$
\begin{aligned}
<f,\phi_i> &= \int_{\Omega}f\phi_i\,\mathrm d\mathrm x\\
& = \int_{\omega_i}f\phi_i\,\mathrm d\mathbf x \\
& = \sum_{\tau\in\omega_i}\int_{\tau}f\phi_i\,\mathrm d\mathbf x\\
& = \sum_{\tau\in\omega_i}\int_{\tau}f\lambda_i\,\mathrm d\mathbf x
\end{aligned}
$$
这意味着, 我们只要在每个三角形单元 $\tau$ 上计算下面三个积分即可, 组装出载荷向量
$$
\begin{aligned}
\int_{\tau}f\lambda_i\,\mathrm d\mathbf x\\
\int_{\tau}f\lambda_j\,\mathrm d\mathbf x\\
\int_{\tau}f\lambda_k\,\mathrm d\mathbf x\\
\end{aligned}
$$

### Dirichlet 边界条件处理

组装出刚度矩阵 $\mathbf A$ 和载荷向量后 $\mathbf f$ 后, 我们还不能直接求解
$$
\mathbf A\mathbf u = \mathbf f
$$
还需要进一步处理边界条件. 

模型问题中, 已经知道 $u$ 在 $\Omega$ 边界上的值, 即边界上的网格节点处的值. 此时, 解向量 $\mathbf u$, 可以分解为两个向量

$$
\mathbf u = \mathbf u_{interior} + \mathbf u_{boundary}
$$

其中

$$
\mathbf u_{interior}[i] = 
\begin{cases}
\mathbf u_i, & \mathbf x_i \text{ 是一个内部点}\\
0, & \mathbf x_i \text{ 是一个边界点}
\end{cases}
$$

$$
\mathbf u_{boundary}[i] = 
\begin{cases}
u(x_i), & \mathbf x_i \text{ 是一个边界点}\\
0, & \mathbf x_i \text{ 是一个内部点}
\end{cases}
$$

则得到的线性代数系统可以做如下的变形:
$$
\begin{aligned}
\mathbf A\mathbf u &= \mathbf f\\
\mathbf A (\mathbf u_{interior} + \mathbf u_{boundary}) &= \mathbf f\\
\mathbf A \mathbf u_{interior} &= \mathbf f - \mathbf A \mathbf u_{boundary}\\
\mathbf A \mathbf u_{interior} &= \mathbf b\\
\end{aligned}
$$
最后一个方程, 取 $ \mathbf b = \mathbf f - \mathbf A \mathbf u_{boundary}$.

由于只需要求内部自由度的值, 上面最后一个方程和右端需要修改一下, 如果 $\mathbf x_i$ 是边界点, 则

1. $\mathbf A$ 的第 $i$ 个主对角元素设为 1, 第 $i$ 行和第 $i$ 列的其它元素都设为 0, 修改后的矩阵记为 $\bar{\mathbf A}$.
1. $\mathbf b$ 的第 $i$ 个分量设为 $u(x_i)$, 修改后的右端向量记为 $\bar{\mathbf b}$.

进而可得线性方程组
$$
\bar{\mathbf A}\mathbf u = \bar{\mathbf b}
$$

注意, 如果 $x_i$ 是边界点, 上述线性方程组中的第 $i$ 个方程, 实际上就是

$$
u_i = u(x_i)
$$


```python

```
