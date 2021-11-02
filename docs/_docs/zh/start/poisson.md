---
title: 任意次有限元求解 Poisson 方程示例
permalink: /docs/zh/start/poisson
key: docs-quick-start-zh
---

# Poisson 方程的标准 Lagrange 有限元方法

给定区域 $\Omega\subset\mbR^d$, 其边界 $\partial \Omega = \Gamma_D \cup \Gamma_N \cup \Gamma_R$

$$
\begin{aligned}
    -\Delta u &= f, \quad\text{in }\Omega\\
    u &= g_D, \quad\text{on }\Gamma_D \leftarrow \text{\bf Dirichlet }\\
    \frac{\partial u}{\partial\bfn} & = g_N, \quad\text{on
    }\Gamma_N\leftarrow \text{\bf Neumann}\\
    \frac{\partial u}{\partial\bfn} + \kappa u& = g_R, \quad\text{on
    }\Gamma_R\leftarrow \text{\bf Robin}
\end{aligned}
$$

其中 
* $\Delta u(x) = u_{xx}$
* $\Delta u(x, y) = u_{xx} + u_{yy}$
* $\Delta u(x, y, z) = u_{xx} + u_{yy} + u_{zz}$
* $\frac{\partial u}{\partial\bfn} = \nabla u\cdot\bfn$

方程两端分别乘以测试函数 $v \in H_{D,0}^1(\Omega)$, 则连续的弱形式可以写为

$$
(f,v) = -(\Delta u, v)
$$

再分部积分

$$
\begin{aligned}
    (f,v)&=-(\Delta u, v)\\
         &=(\nabla u, \nabla v)-<\nabla u \cdot \bfn,v>_{\partial\Omega}\\
         &=(\nabla u,\nabla v)-<g_N,v>_{\Gamma_N}
         +<\kappa u,v>_{\Gamma_R}-<g_R,v>_{\Gamma_R}
\end{aligned}
$$

整理可得

$$
(\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_R} = 
(f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$
其中测试函数 $v|_{\Gamma_D} = 0$!

取一个 $N$ 维的有限维空间 $V_h = \ospan\{\phi_i\}_0^{N-1}$，其基函数向量记为

$$
\boldsymbol \phi = [\phi_0, \phi_1, \cdots, \phi_{N-1}],
$$

用 $V_h$ 替代无限维的空间 $H^1_{D,0}(\Omega)$, 从而把问题转化为{\bf
离散的弱形式}：求 $u_h = \bphi\bfu = \sum_{i=0}^{N-1}u_i\phi_i\in V_h$, 对任意 $\phi_i$, 满足：

$$
(\nabla u_h,\nabla \phi_i)+<\kappa u_h, \phi_i>_{\Gamma_R}= (f, \phi_i)+<g_R, \phi_i>_{\Gamma_R}+<g_N, \phi_i>_{\Gamma_N},
$$

其中 $\boldsymbol u$ 是 $u_h$ 在基函数 $\bphi$ 下的坐标{\bf 列向量}, 即 $\boldsymbol =[u_0, u_1, \ldots, u_{N-1}]^T$。

最终可以获得离散的代数系统
\begin{equation*}
    (\bfA + \bfR)\bfu = \bfb + \bfb_N+ \bfb_R
\end{equation*}
\begin{align*}
    \bfA =& \int_\Omega (\nabla \bphi)^T \nabla\bphi\rmd\bfx 
    = \sum_{\tau\in\mcT} \int_\tau (\nabla \bphi|_\tau)^T \nabla\bphi|_\tau\rmd\bfx\\ 
    \bfR =& \int_{\Gamma_R} \bphi^T \bphi\rmd\bfs 
    = \sum_{e_R\in\Gamma_R}\int_{e_R} (\bphi|_{e_R})^T \bphi|_{e_R}\rmd\bfs\\ 
    \bfb =& \int_\Omega f\bphi^T\rmd\bfx 
    = \sum_{\tau\in\mcT}\int_\tau f (\bphi|_\tau)^T\rmd\bfx \\
    \bfb_N = & \int_{\Gamma_N} g_N\bphi^T\rmd\bfs 
    = \sum_{e_N\in\Gamma_N}\int_{e_N}g_N(\bphi|_{e_N})^T\rmd\bfs \\
    \bfb_R = & \int_{\Gamma_R} g_R\bphi^T\rmd\bfs 
    = \sum_{e_R\in\Gamma_R}\int_{e_R}g_R(\bphi|_{e_R})^T\rmd\bfs
\end{align*}
\end{frame}



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

