---
title: Nedelec 有限元空间 
permalink: /docs/zh/space/nedelec-space
key: docs-nedelec-space-zh
---


记 $K$ 为一三角形单元, 三条边分别记为 $F_0$, $F_1$ 和 $F_2$, 设 $\boldsymbol m_k$ 是定义
在 $K$ 上 $k$ 次缩放单项式空间的基函数向量组, 共有 $n_k:=(k+1)(k+2)/2$ 个函数组
成. 进一步设 $\boldsymbol f_{k}^F$ 为边 $F$ 上的 $k$ 次多项式空间基函数向量组,
共有 $k+1$ 个基函数.

$$
\boldsymbol M_k = 
    \begin{bmatrix}
        \boldsymbol m_{k} & \boldsymbol 0 \\ 
        \boldsymbol 0 & \boldsymbol m_{k} 
    \end{bmatrix} 
$$

$$
\nabla \boldsymbol M_k = 
\begin{bmatrix}
\end{bmatrix}
$$

$$
\boldsymbol \Phi_k = 
    \begin{bmatrix}
        \boldsymbol M_k & [m_2, -m_1]^T \bar{\boldsymbol m}_k\\ 
    \end{bmatrix} 
$$
其中 $\bar{\boldsymbol m}_k$ 是所有的 $k$ 次齐次缩放单项式基函数向量.

$K$ 上的 Nedelec 元的基函数可以写成如下形式:

$$
\boldsymbol \phi(\boldsymbol x) = \boldsymbol \Phi_{k} \boldsymbol c 
$$
其中 $\boldsymbol c$ 是长度为 $(k+1)(k+3)$ 的列向量.

逆时针旋转边的法向 $\boldsymbol t = [-n_1, n_0]^T$


$$
\begin{aligned}
    \begin{bmatrix}
        t_{0}^{F_0} \int_{F_0} \left(\boldsymbol f_{k}^{F_0}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        t_{1}^{F_0} \int_{F_0} \left(\boldsymbol f_{k}^{F_0}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        \int_{F_0} 
        \left(t_{0}^{F_0}m_2 - t_{1}^{F_0} m_1\right)
        \left(\boldsymbol f_{k}^{F_0}\right)^T\bar\boldsymbol m_{k}\mathrm d\boldsymbol s \\ 
        t_{0}^{F_1} \int_{F_1} \left(\boldsymbol f_{k}^{F_1}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        t_{1}^{F_1} \int_{F_1} \left(\boldsymbol f_{k}^{F_1}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        \int_{F_1} 
        \left(t_{0}^{F_1}m_2 - t_{1}^{F_1}m_1\right)
        \left(\boldsymbol f_{k}^{F_1}\right)^T\bar\boldsymbol m_{k}\mathrm d\boldsymbol s \\ 
        t_{0}^{F_2} \int_{F_2} \left(\boldsymbol f_{k}^{F_2}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        t_{1}^{F_2} \int_{F_2} \left(\boldsymbol f_{k}^{F_2}\right)^T\boldsymbol m_{k}\mathrm d\boldsymbol s & 
        \int_{F_2} 
        \left(t_{0}^{F_2}m_2 - t_{1}^{F_2}m_1\right)\left(\boldsymbol f_{k}^{F_2}\right)^T\bar\boldsymbol m_{k}\mathrm d\boldsymbol s \\ 
        \int_{K} \boldsymbol m_{k-1}^T\boldsymbol m_{k}\mathrm d\boldsymbol x & \boldsymbol 0 & 
        \int_{K} m_2\boldsymbol m_{k-1}^T\bar\boldsymbol m_{k} \mathrm d\boldsymbol x \\ 
        \boldsymbol 0 & \int_{K} \boldsymbol m_{k-1}^T\boldsymbol m_{k}\mathrm d\boldsymbol x & 
        -\int_{K} m_1\boldsymbol m_{k-1}^T\bar\boldsymbol m_{k} \mathrm d\boldsymbol x \\ 
    \end{bmatrix}
\end{aligned}
$$
