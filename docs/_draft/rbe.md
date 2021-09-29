---
title: 结构力学有限元分析中的 RBE 单元 
tags: FEALPy
---

设给定一个结构体，四面体网格上的离散系统为

$$
KU = F
$$

网格中的节点分为四类

1. 位移边界节点，给定位移的网格节点，一般是固定的，即位移为 0。
1. 自由节点，其中的位移未知。
1. 集中节点，其位移给定或者自由，或者给定了力的条件。
1. 依赖节点，其位移由参考节点决定。

设 $U_0$ 是自由节点对应的自由度向量，$U_1$ 是依赖节点对应的自由度向量，$U_2$
是集中节点对应的自由度向量。

$$
\begin{bmatrix}
K_{0,0} & K_{0,1} \\
K_{1,0} & K_{1,1} \\
\end{bmatrix}
\begin{bmatrix}
U_0 \\ U_1
\end{bmatrix}
=
\begin{bmatrix}
F_0 \\ F_1
\end{bmatrix}
$$

其中 $K_{1, 0} = K_{0, 1}^T$, 且依赖节点的自由度向量 $U_1$
是依赖集中节点的自由度向量 $U_2$，关系如下

$$
U_1 = GU_2
$$

则原来的方程可以转化为

$$
\begin{bmatrix}
K_{0,0} & K_{0,1}G \\
G^T K_{1,0} & G^TK_{1,1}G \\
\end{bmatrix}
\begin{bmatrix}
U_0 \\ U_2 
\end{bmatrix}
=
\begin{bmatrix}
F_0 \\ G^TF_1
\end{bmatrix}
$$



