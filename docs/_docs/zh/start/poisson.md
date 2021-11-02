---
title: 任意次有限元求解 Poisson 方程示例
permalink: /docs/zh/start/poisson
key: docs-quick-start-zh
---

# Poisson 方程的标准 Lagrange 有限元方法

$\quad$ 给定区域 $\Omega\subset\mathbb R^d$, 其边界 $\partial \Omega = \Gamma_D \cup \Gamma_N \cup \Gamma_R$.
经典的 Poisson 方程形式如下(方便起见, 我们称其为 **A 问题**吧)

$$
\begin{aligned}
    -\Delta u &= f, \quad\text{in }\Omega\\
    u &= g_D, \quad\text{on }\Gamma_D \leftarrow \text{\bf Dirichlet }\\
    \frac{\partial u}{\partial\boldsymbol n} & = g_N, \quad\text{on
    }\Gamma_N\leftarrow \text{\bf Neumann}\\
    \frac{\partial u}{\partial\boldsymbol n} + \kappa u& = g_R, \quad\text{on
    }\Gamma_R\leftarrow \text{\bf Robin}
\end{aligned}
$$

其中 
* 一维情形：$\Delta u(x) = u_{xx}$
* 二维情形：$\Delta u(x, y) = u_{xx} + u_{yy}$
* 三维情形：$\Delta u(x, y, z) = u_{xx} + u_{yy} + u_{zz}$
* $\frac{\partial u}{\partial\boldsymbol n} = \nabla u\cdot\boldsymbol n$

对于绝大多数偏微分方程来说, 往往都找不到**解析形式**的解. 因此在实际应用问题当中,
**数值求解偏微分方程**才是可行的手段.
但这里要解决**偏微分方程方程的无限性**和**计算资源的有限性**这一对本质矛盾.
这里的**无限性**有两个方面的含义, 一个是方程的解的无限性, 即要求区域 $\Omega$ 中
**无穷多个点处的函数值**; 另一个是函数导数定义的无限性. 而我们所用的求解工具
**计算机**, 从存储和计算速度上来说, 都是有限的.
如何克服偏微分方程数值解的无限性, 设计出可以在计算机上高效运行的算法,
是**偏微分方程数值解**的主要研究内容. 下面我们着重介绍有限元这一类方法是如何解决这个无限性的问题的. 

$\quad$ 由于在原来方程的形式下, 根本无法找到克服无限性的办法,
所以必须寻找新的方程形式. 有限元方法采用办法就是**变分**. 

$\quad$ 首先引入一个无限维的函数空间

$$
H_{D,0}^1(\Omega) := \{ v\in L^2(\Omega): \nabla v \in L^2(\Omega;\mathbb R^d), v|_{\Gamma_D} = 0\}
$$

其中 $L^2(\Omega)$ 是**平方可积**的标量函数空间, $L^2(\Omega;\mathbb R^d)$
表示每个分量都平方可积的 $d$ 维向量函数空间.

$\quad$ 然后在方程两端分别乘以任意的函数 $v \in H_{D,0}^1(\Omega)$(称为测试函数), 

$$
(f,v) = -(\Delta u, v),
$$

再分部积分

$$
\begin{aligned}
    (f,v)&=-(\Delta u, v)\\
         &=(\nabla u, \nabla v)-<\nabla u \cdot \boldsymbol n,v>_{\partial\Omega}\\
         &=(\nabla u,\nabla v)-<g_N,v>_{\Gamma_N}
         +<\kappa u,v>_{\Gamma_R}-<g_R,v>_{\Gamma_R}
\end{aligned}
$$

整理可得一个**积分**形式的方程(称为原方程的**连续的弱形式**, 我们称其为 **B 问题**)

$$
(\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_R} = 
(f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

注意, 因为测试函数在 Dirichlet 边界上取值为 0, 即 $v|_{\Gamma_D} = 0$, 所以
$\Gamma_D$ 上积分项消失了. 上面的推导过程, 把一个逐点意义下成立的方程转化为了一个积分形式的方程,
它对任意的 $v \in H_{D,0}^1(\Omega)$ 都成立.

$\quad$ 通过上面的**变分过程**把 **A 问题**变成了 **B 问题**, 里面的动机是什么?
当然是想把原来无法解决的问题变成一个可以解决的问题, 或者说变成一个更容易解决的问题.
那么 **B 问题**可以求解或者更容易求解了吗? 很容易看到, **B 问题**还是不可以直接求解,
因为空间 $H_{D,0}^1(\Omega)$ 是无限维, 取遍所有的 $v$,
可以得到无穷多个积分方程. 但相比于原来的方程, 解也不在要求逐点存在了, 涉及的二阶导数也变成了一阶导数, 所以可以说问题的难度降低了.

$\quad$ 当然, 把 **A 问题**转化为 **B 问题**, 还有一个重要的理论问题要回答, 即 **B
问题**还和原来的 **A 问题**等价吗? 当然在一定条件下, 它们仍然是等价的, 这由经典的偏微分方程理论来保证, 这里不赘述.

$\quad$ 更为重要的是, 方程形式的变化为我们提供了一条从无限走向有限的新途径.
对于**连续的弱形式**方程来说, 核心问题还是无限性. 办法只有一个,
用有限维的空间替代无限维的空间 $H_{D,0}^1(\Omega)$.  

$\quad$ 这里先不讨论有限维空间如何构造, 这是编程要解决的核心问题, 后面一系列文章会详细展开. 
这里我们先假设有一个 $N$ 维的有限维空间 $V_h = \operatorname{span}\{\phi_i\}_0^{N-1}$, 
并把**基函数**组成的向量记为

$$
\boldsymbol \phi = [\phi_0, \phi_1, \cdots, \phi_{N-1}],
$$

注意, 这里我们约定  $\boldsymbol \phi$ 是一个**行向量**. $\boldsymbol \phi$ 的梯度记为

$$
\nabla \boldsymbol \phi = [\nabla \phi_0, \nabla \phi_1, \cdots, \nabla \phi_{N-1}],
$$

这里标量函数的梯度默认是**列向量**的形式. 那么 $\nabla\boldsymbol\phi$
实际上是一个形状为 $N\times d$ 的矩阵函数.

$\quad$ 用 $V_h$ 替代无限维的空间 $H^1_{D,0}(\Omega)$,
并假设要找的解 $u$ 也在这个空间, 重新记为 $u_h$, 可以表示为如下形式

$$
u_h = \boldsymbol \phi\boldsymbol u = \sum_{i=0}^{N-1}u_i\phi_i\in V_h,
$$

其中 $\boldsymbol u$ 是 $u_h$ 在基函数 $\boldsymbol\phi$ 下的**坐标向量**, 或者
称为**自由度向量**, 即 

$$
\boldsymbol u =
\begin{bmatrix}
u_0 \\ u_1\\ \vdots \\ u_{N-1}
\end{bmatrix}.
$$

注意这里 $\boldsymbol u$ 是**列向量**.  

我们得到一个新的问题形式, 即**离散的弱形式**: 求  $u_h$,  满足

$$
(\nabla u_h,\nabla v_h)+<\kappa u_h, v_h>_{\Gamma_R}= (f, v_h)+<g_R, v_h>_{\Gamma_R}+<g_N, v_h>_{\Gamma_N}, 
\quad\forall v_h \in V_h,
$$

这里顺其自然称其为我们的 **C 问题**吧.  表面上 $V_h$ 中仍然有无穷多个 $v_h$, 
但实际上离散的弱形式只需要所有的基函数 $\boldsymbol\phi$ 成立即可. 进一步,
我们可以用矩阵向量的形式重新改写一下这个**离散的弱形式**, 即用
$\boldsymbol\phi\boldsymbol u$ 替换 $u_h$, $\boldsymbol \phi$ 替换 $v_h$,
并写成显式积分的形式 

$$
\int_\Omega (\nabla \boldsymbol \phi)^T \nabla\boldsymbol \phi\mathrm d\boldsymbol x\cdot\boldsymbol u +
\int_{\Gamma_R} \kappa\boldsymbol \phi^T \boldsymbol \phi\mathrm d\boldsymbol s\cdot
\boldsymbol u = 
\int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x + 
\int_{\Gamma_R} g_R\boldsymbol \phi^T\mathrm d\boldsymbol s + 
\int_{\Gamma_N} g_N\boldsymbol \phi^T\mathrm d\boldsymbol s
$$

最终可以获得离散的代数系统

$$
(\boldsymbol A + \boldsymbol R)\boldsymbol u = \boldsymbol b + \boldsymbol b_N+ \boldsymbol b_R
$$

$$
\begin{aligned}
    &\boldsymbol A = \int_\Omega (\nabla \boldsymbol \phi)^T \nabla\boldsymbol \phi\mathrm d\boldsymbol x, \quad 
    \boldsymbol R = \int_{\Gamma_R} \boldsymbol \phi^T \boldsymbol \phi\mathrm d\boldsymbol s \\
    &\boldsymbol b = \int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x,
    \quad  
    \boldsymbol b_N =  \int_{\Gamma_N} g_N\boldsymbol \phi^T\mathrm d\boldsymbol s,  
    \quad
    \boldsymbol b_R =  \int_{\Gamma_R} g_R\boldsymbol \phi^T\mathrm d\boldsymbol s 
\end{aligned}
$$

$$
\begin{aligned}
    \boldsymbol A =& \int_\Omega (\nabla \boldsymbol \phi)^T \nabla\boldsymbol \phi\mathrm d\boldsymbol x 
    = \sum_{\tau\in\mathcal T} \int_\tau (\nabla \boldsymbol \phi|_\tau)^T \nabla\boldsymbol \phi|_\tau\mathrm d\boldsymbol x\\ 
    \boldsymbol R =& \int_{\Gamma_R} \boldsymbol \phi^T \boldsymbol \phi\mathrm d\boldsymbol s 
    = \sum_{e_R\in\Gamma_R}\int_{e_R} (\boldsymbol \phi|_{e_R})^T \boldsymbol \phi|_{e_R}\mathrm d\boldsymbol s\\ 
    \boldsymbol b =& \int_\Omega f\boldsymbol \phi^T\mathrm d\boldsymbol x 
    = \sum_{\tau\in\mathcal T}\int_\tau f (\boldsymbol \phi|_\tau)^T\mathrm d\boldsymbol x \\
    \boldsymbol b_N = & \int_{\Gamma_N} g_N\boldsymbol \phi^T\mathrm d\boldsymbol s 
    = \sum_{e_N\in\Gamma_N}\int_{e_N}g_N(\boldsymbol \phi|_{e_N})^T\mathrm d\boldsymbol s \\
    \boldsymbol b_R = & \int_{\Gamma_R} g_R\boldsymbol \phi^T\mathrm d\boldsymbol s 
    = \sum_{e_R\in\Gamma_R}\int_{e_R}g_R(\boldsymbol \phi|_{e_R})^T\mathrm d\boldsymbol s
\end{aligned}
$$



\begin{frame}
  \frametitle{单纯形网格}
  有限元方法构造有限维空间的方法，首先是把求解区域离散成网格，即很多{\bf
  简单几何区域}组成的集合，如下面的{\bf 几何单纯形网格}：
  \begin{figure}
    \centering
    \includegraphics[scale=0.2]{./figures/interval.png}
    \includegraphics[scale=0.2]{./figures/triholemesh.png}
    \includegraphics[scale=0.2]{./figures/tetboxmesh.png}
    \caption{区间, 三角形和四面体网格.}
  \end{figure}
  \begin{remark}
    \begin{itemize}
      \item[$\bullet$] 各种有限元方法的核心问题是如何构造有限维的子空间 $V_h$!
      \item[$\bullet$] 本节将讨论求解区域 $\Omega$ 离散为{\bf 单纯形网格}后,
        经典的 Lagrange 有限元空间的构造及程序实现. 
    \end{itemize}
  \end{remark}
\end{frame}

