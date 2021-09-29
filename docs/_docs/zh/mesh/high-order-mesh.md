---
title: 高阶有限元网格 
permalink: /docs/zh/mesh/high-order-mesh
key: docs-high-order-mesh-zh
---

# 1. 数学基础

设 $\boldsymbol x$ 是曲面 $S$ 上的一点, 且曲面 $S$ 有参数表示 $\boldsymbol x(\xi,\eta)
= [x(\xi,\eta),y(\xi,\eta),z(\xi,\eta)]$, 记 $\boldsymbol n$ 为曲面 $S$ 的
单位法向量, 则有

$$
\boldsymbol n = \frac{\boldsymbol x_{\xi} \times \boldsymbol x_{\eta}}{\| \boldsymbol x_{\xi} \times \boldsymbol x_{\eta} \|}
$$

曲面 $S$ 的**第一基本形式**对应的二次微分式:

$$
I = <\mathrm d\boldsymbol x, \mathrm d\boldsymbol x> = g_{00}\mathrm d\xi d\xi + 2g_{01}\mathrm d\xi \mathrm d\eta + g_{11}\mathrm d\eta d\eta
$$

其中 $\mathrm d \boldsymbol x = \boldsymbol x_{\xi}\mathrm d\xi+ \boldsymbol x_{\eta}\mathrm d\eta$,

$$
g_{00} = <\boldsymbol x_{\xi}, \boldsymbol x_{\xi}>, 
g_{01} = <\boldsymbol x_{\xi}, \boldsymbol x_{\eta}>,
g_{11} = <\boldsymbol x_{\eta}, \boldsymbol x_{\eta}>,
$$

称为曲面 $S$ 第一基本形式 $I$ 的系数。

> **注意**
>    * 曲面的第一基本形式是不依赖于曲面的具体参数化.
>    * 曲面的第一基本形式在$\mathbb R^3$的合同变化下不变.

曲面 $S$ 的**第二基本式**对应的二次微分式为

$$
II = -<\mathrm d\boldsymbol x,\mathrm d\boldsymbol n> = 
b_{00}\mathrm d\xi \mathrm d\xi + 
2b_{01}\mathrm d\xi \mathrm d\eta + 
b_{11}\mathrm d\eta \mathrm d\eta
$$

其中 $\mathrm d\boldsymbol n = \boldsymbol n_{\xi} \mathrm d\xi + \boldsymbol n_{\eta}\mathrm d\eta$.

$$
\begin{aligned}
b_{00} = <\boldsymbol x_{\xi \xi},\boldsymbol n> &= 
-<\boldsymbol x_{\xi},\boldsymbol n_{\xi}> \\
b_{01} = <\boldsymbol x_{\xi \eta},\boldsymbol n> &= 
-<\boldsymbol x_{\xi},\boldsymbol n_{\eta}> =
- <\boldsymbol x_{\eta},\boldsymbol n_{\xi}> \\
b_{11} = <\boldsymbol x_{\eta \eta},\boldsymbol n> &= 
-<\boldsymbol x_{\eta},\boldsymbol n_{\eta}>
\end{aligned}
$$

称为曲面 $S$ 第二基本形式 $II$ 的系数.

第二基本形式具有如下性质

> * 设 $\boldsymbol x = \boldsymbol x(\xi,\eta)$ 和 $\boldsymbol x = \boldsymbol x(\overline{\xi},
> \overline{\eta})$ 是曲面 $S$ 的两个不同的参数表示，当变换
> $(\xi,\eta) \rightarrow (\overline{\xi}, \overline{\eta})$ 的雅克比矩阵的行列式
> 大于零，第二基本形式不变; 当变换 $(\xi,\eta) \rightarrow (\overline{\xi},
> \overline{\eta})$ 的雅克比矩阵的行列式小于零，第二基本形式改变符号.
> * 设 $S$ 是曲面 $\mathbb R^3$ 中的一张曲面，$\boldsymbol x(\xi,\eta)$ 是它的参数表述; $\mathcal T$
> 是 $\mathbb R^3$ 的一个合同变换，则曲面 $\tilde S: \mathcal T \circ \boldsymbol x(\xi,\eta)$ 
> 的第二基本形式 $\widetilde{II}$ 与曲面 $S$ 的 第二基本形式有如下关系:
> 当 $\mathcal T$ 为刚体运动(合同变换分解的正交变换的矩阵的行列式为1)时 
> $\widetilde{II}(\xi,\eta) = \widetilde(\xi,\eta)$, 当 $\mathcal T$ 为反向刚
> 体运动(合同变换分解的正交变换的矩阵的行列式为-1)时
> $\widetilde{II}(\xi,\eta) = -II(\xi,\eta)$.

# 2. 高阶单元

## 2.1 高阶一维单元

$\qquad$ 给定 $p$ 次拉格朗日曲线 $l$ 上有 $p+1$ 个插值节点 
$[\boldsymbol x_0, \boldsymbol x_1, \cdots, \boldsymbol x_p]$. 
参考单元为 $[0, 1]$, 坐标变量记为 $u$.  曲线 $l$ 相对于参考单元的重心坐标函数是

$$
\lambda_0 = 1-u, \lambda_1 = u,
$$

基于重心坐标函数, 可以构造任意 $p$ 次的**拉格朗日形函数**

$$
\boldsymbol\phi(u) = [\phi_0(u), \phi_1(u), \cdots, \phi_p(u)],
$$

这里的**形函数**和前面的插值点一起决定的曲线 $l$ 的**形状**.

$\qquad$ 对于任意的 $\boldsymbol x \in l$, 存在唯一的 $u \in [0, 1]$, 使得

$$
\boldsymbol x(u) = \boldsymbol x_0 \phi_0(u) + 
\cdots + \boldsymbol x_p\phi_p(u),
$$

进一步, $\boldsymbol x(u)$ 关于 $u$ 的导数为

$$
\boldsymbol x_u = \boldsymbol x_0 (\phi_0)_u + \cdots + \boldsymbol x_p(\phi_p)_u
$$

$\qquad$ 在曲线 $l$ 上可定义 $q$ 次的拉格朗日多项式空间，其所有基函数组成的向量函数记为

$$
\boldsymbol\varphi(\boldsymbol x) = [\varphi_0(\boldsymbol x), 
\varphi_1(\boldsymbol x), \cdots, \varphi_q(\boldsymbol x)].
$$

它关于 $\boldsymbol x$ 的梯度记为

$$
\nabla_{\boldsymbol x}\boldsymbol\varphi = 
[
\nabla_{\boldsymbol x}\varphi_0, 
\nabla_{\boldsymbol x}\varphi_1, 
\cdots, 
\nabla_{\boldsymbol x}\varphi_q].
$$

注意这里的梯度是列向量, 这些基函数对应的插值点记为

$$
[\boldsymbol y_0, \boldsymbol y_1, \cdots, \boldsymbol y_q]
$$

其中 $\boldsymbol y_0 = \boldsymbol x_0$, $\boldsymbol y_q = \boldsymbol x_p$.
注意这里 $q$ 可以与 $p$ 不同.

$\qquad$ 对于任意的 $\boldsymbol x \in l$, 都有唯一的一个 $u \in [0, 1]$ 
和它对应, 则

$$
\boldsymbol\varphi(\boldsymbol x) = \boldsymbol\varphi(\boldsymbol x(u))
$$

也可以看成定义在 $[0, 1]$ 区间上的函数.


$\qquad$ 下面考虑 $q$ 次的基函数向量 $\boldsymbol \varphi$ 关于 
$\boldsymbol x \in l$ 的导数计算问题

$$
\nabla_{\boldsymbol x} \boldsymbol\varphi = \boldsymbol x_u \boldsymbol G^{-1} 
\boldsymbol\varphi_u
$$

其中 $\boldsymbol G = <\boldsymbol x_u, \boldsymbol x_u>$.

## 2.2 高阶三角形单元

$\qquad$ 给定一个 $p$ 次曲面三角形 $\tau$,  其上有 $n = (p+1)(p+2)/2$ 
插值点, 记为

$$
[\boldsymbol x_0, \boldsymbol x_1, \cdots, \boldsymbol x_{n}].
$$

取 $\tau$ 对应的参考单元为单位等腰直角三角形, 记为 $\tilde\tau$, 对应的参考坐标记为 
$\boldsymbol u = (u, v)$, 则对应的**重心坐标**为 

$$
\lambda_0 = 1 - u - v,\quad \lambda_1 = u,\quad \lambda_2 = v.
$$

其关于 $\boldsymbol u$ 的梯度为

$$
\nabla_\boldsymbol u \lambda_0 = 
\begin{bmatrix}
    -1 \\ -1
\end{bmatrix},\quad
\nabla_\boldsymbol u \lambda_1 = 
\begin{bmatrix}
    1 \\ 0
\end{bmatrix}, \quad
\nabla_\boldsymbol u \lambda_2 = 
\begin{bmatrix}
    0 \\ 1
\end{bmatrix}
$$

基于重心坐标, 可以构造定义在 $\tilde\tau$ 上的 $n$ 个 $p$ 次的拉格朗日形函数, 
组成的向量函数记为

$$
\boldsymbol \phi(u, v) = [\phi_0(u, v), \phi_1(u, v), \cdots, \phi_n(u, v)].
$$

则 $p$ 次曲面 $\tau$ 上的任意一点 $\boldsymbol x$ 可表示为

$$
\boldsymbol x = \boldsymbol x_0 \phi_0 + \boldsymbol x_1\phi_1 + 
\cdots + \boldsymbol x_n\phi_n,
$$

则 $\boldsymbol x$ 关于 $\boldsymbol u$ 的 Jacobi 矩阵为

$$
\nabla_\boldsymbol u \boldsymbol x = 
\boldsymbol x_0 \nabla_\boldsymbol u^T \phi_0 + 
\boldsymbol x_1\nabla_\boldsymbol u^T \phi_1 + 
\cdots + 
\boldsymbol x_n\nabla_\boldsymbol u^T\phi_n
$$

进一步 $\tau$ 上的第一基本形式为：

$$
I  = 
\begin{bmatrix}
    \mathrm d u , \mathrm dv
\end{bmatrix}
\boldsymbol G
\begin{bmatrix}
    \mathrm du \\ \mathrm dv
\end{bmatrix} = 
\begin{bmatrix}
    \mathrm d u , \mathrm dv
\end{bmatrix}
 \begin{bmatrix}
    g_{00} & g_{01} \\
    g_{10} & g_{11} 
\end{bmatrix}   
\begin{bmatrix}
    \mathrm du \\ \mathrm dv
\end{bmatrix} = 
g_{00} \mathrm du \mathrm d u +
2g_{01} \mathrm d u \mathrm d v + 
g_{11} \mathrm d v \mathrm d v.
$$

其中

$$
\begin{aligned}
g_{00} = & <\boldsymbol x_\xi, \boldsymbol x_\xi>, \\
g_{01} = & <\boldsymbol x_\xi, \boldsymbol x_\eta> = g_{10},\\
g_{11} = & <\boldsymbol x_\eta, \boldsymbol x_\eta>.
\end{aligned}
$$

马上可得 $p$ 次曲面 $\tau$ 上的切梯度算子：

$$
\nabla_\tau \boldsymbol \phi = 
\nabla_\boldsymbol u \boldsymbol x \boldsymbol G^{-1} 
\nabla_\boldsymbol u \boldsymbol \phi.
$$


也可得 $\tau$ 的面积计算公式为:

$$
|\tau| = \int_\tau 1 \mathrm d\boldsymbol x =
\int_{\bar \tau} |\boldsymbol x_u \times \boldsymbol x_v| 
\mathrm d\boldsymbol u = 
\int_{\bar \tau} \sqrt{|\boldsymbol G|}
\mathrm d\boldsymbol u.
$$

上面用到了如下的关系式

$$
|\boldsymbol x_u \times \boldsymbol x_v|^2 = 
|\boldsymbol x_u|^2 |\boldsymbol x_v|^2  \cos^2 \theta  = 
|\boldsymbol x_u|^2 |\boldsymbol x_v|^2  -
|\boldsymbol x_u|^2 |\boldsymbol x_{v}|^2 \sin^2 \theta = 
|\boldsymbol G |.
$$

在 $p$ 次三角形曲面 $\tau$ 上可定义 $q$ 次的拉格朗日多项式空间
(注意这里不要求 $q=p$)， 其基函数个数为 $m=(q+1)(q+2)/2$, 记为向量形式

$$
\boldsymbol \varphi(\boldsymbol x) = [\varphi_0, \varphi_1, \cdots, \varphi_m]
$$

对应的插值点记为

$$
[\boldsymbol y_0, \boldsymbol y_1, \cdots, \boldsymbol y_m]
$$

满足如下插值性质

$$
\varphi_i(\boldsymbol y_j) = 
\begin{cases}
1, & i = j \\
0, & i\not=j
\end{cases}
$$

则 $\boldsymbol\varphi$ 关于 $\boldsymbol x$ 的梯度计算公式为

$$
\nabla_{\boldsymbol x} \boldsymbol \varphi = 
\boldsymbol x_{\boldsymbol u}\boldsymbol G^{-1}
\nabla_{\boldsymbol u} \varphi
$$

Hessian 矩阵的计算公式为

$$
\nabla^2_\boldsymbol x \boldsymbol\varphi = 
\nabla^2_u \boldsymbol x (\boldsymbol G^{-1})^2 \nabla_u \boldsymbol\varphi + 
\nabla_u \boldsymbol x (\boldsymbol G^{-1})^2 
\nabla_{uu}\boldsymbol\varphi\nabla_u \boldsymbol x.
$$

$p$ 次曲面 $\tau$ 上的第二基本形式为：

$$
II  =
\begin{bmatrix}
    \mathrm d u , \mathrm d v
\end{bmatrix}
 \begin{bmatrix}
    b_{00} & b_{01} \\
    b_{10} & b_{11} 
\end{bmatrix}   
\begin{bmatrix}
    \mathrm du \\ \mathrm dv
\end{bmatrix} = 
< \mathrm d\boldsymbol x, \mathrm d\boldsymbol n > =
b_{00} \mathrm du \mathrm d u + 
2b_{01} \mathrm d u \mathrm d v +
b_{11} \mathrm d v \mathrm d v. 
$$

其中

$$
\begin{aligned}
    \boldsymbol n =& \frac{\boldsymbol x_{u} \times \boldsymbol x_{v}}
    {\|\boldsymbol x_{u} \times \boldsymbol x_{v}\|},\\
	\mathrm d\boldsymbol x = & \boldsymbol x_u\mathrm du +\boldsymbol x_v\mathrm dv, \\
	\mathrm d\boldsymbol n = & \boldsymbol n_u\mathrm du + \boldsymbol n_v\mathrm dv, \\
	b_{00} = & <\boldsymbol x_{u u},\boldsymbol n> = 
    -<\boldsymbol x_u,\boldsymbol n_u>, \\
	b_{01} = & <\boldsymbol x_{u v},\boldsymbol n> = 
    -<\boldsymbol x_u,\boldsymbol n_v> = 
    - <\boldsymbol x_v,\boldsymbol n_u> = b_{10},\\
	b_{11} = & <\boldsymbol x_{v v},\boldsymbol n> = 
    -<\boldsymbol x_v,\boldsymbol n_v>.
\end{aligned}
$$

## 2.3 高阶四面体单元

## 2.4 高阶四边形单元

$\qquad$ 高阶四边形单元，可以通过两个高阶一维单元做张量积的方式构造.

## 2.5 高阶六面体单元

$\qquad$ 高阶六面体单元，可以通过三个高阶一维单元做张量积的方式构造.

## 2.6 高阶三棱柱单元

$\qquad$ 高阶三棱柱单元, 可以通过高阶三角形单元和高阶一维单元做用张量积的方式构造.
