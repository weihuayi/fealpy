---
title: 高次插值
tags: interpolation
author: wx
---


给定一个三角形单元 
$\tau := (\boldsymbol x_0, \boldsymbol x_1, \boldsymbol x_2)$, 

TODO: 加个图

一致加密后, 得到四个子单元

TODO: 图

$$
\tau_0 = \\
\tau_1 = \\
\tau_2 = \\
\tau_3 = \\
$$

下面讨论给定子单元 $\tau_i$ 中的一个点的重心坐标 $\lambda_{i0}$, $\lambda_{i1}$, 
$\lambda_{i2}$, 如何计算出该点在父单元 $\tau$ 中的重心坐标.

$$
\boldsymbol  
$$


<img src="../assets/images/interploation/refine.png" alt="refine" style="zoom:50%;" />

- 首先小单元内`\bf y` 都可以写成小单元三个顶点的线性组合形式， 即

$$
\bf y = \lambda_0 \bf y_0 + \lambda_1 \bf y_1 + \lambda_2 \bf y_2,
$$

其中 `\lambda_0, \lambda_1, \lambda_2` 是点 `\bf y` 的重心坐标，且 `\lambda_0 + \lambda_1 + \lambda_2 = 1`， `\lambda_i \geq 0, i=0,1,2`。

<img src="../assets/images/interploation/bcs.png" alt="refine" style="zoom:100%;" />

几何角度上， 重心坐标可以写成面积的形式，即

$$
\begin{aligned}
 \lambda_0 &= \frac{S_0}{S_0+S_1+S_2} \\
\lambda_1 &= \frac{S_1}{S_0+S_1+S_2} \\
\lambda_2 &= \frac{S_2}{S_0+S_1+S_2}
\end{aligned}
$$

同理， 因为`\bf y_0, \bf y_1, \bf y_2` 是大单元即三角形`\triangle_{{\bf x_0}{\bf x_1}{\bf x_2}}`内的点，所以同样可以用重心坐标的形式得到

$$
\begin{aligned}
\bf y_0 &= \xi^0_0 \bf x_0 + \xi^0_1 \bf x_1 + \xi^0_2 \bf x_2,\\
\bf y_1 &= \xi^1_0 \bf x_0 + \xi^1_1 \bf x_1 + \xi^1_2 \bf x_2, \\
\bf y_2 &= \xi^2_0 \bf x_0 + \xi^2_1 \bf x_1 + \xi^2_2 \bf x_2,
\end{aligned}
$$

其中 `\xi^{j}_0, \xi^{j}_1, \xi^{j}_2` 是点 `\bf y_{j}` 的重心坐标，且 `\xi^{j}_0 + \xi^{j}_1 + \xi^{j}_2 = 1`， `\xi^{j}_i \geq 0, i, j= 0,1,2`。
重心坐标`\bf \xi`也可以写成面积的形式，

<img src="../assets/images/interploation/bc1.png" alt="refine" style="zoom:50%;" />

$$
\begin{aligned}
 \xi^0_0 &= \frac{M_0}{M_0+M_1+M_2} = \frac{M_0}{M_0+M_1} \\
\xi^0_1 &= \frac{M_1}{M_0+M_1+M_2}=\frac{M_1}{M_0+M_1} \\
\xi^0_2 &= \frac{M_2}{M_0+M_1+M_2} =0
\end{aligned}
$$

<img src="../assets/images/interploation/bc2.png" alt="refine" style="zoom:50%;" />

$$
\begin{aligned}
 \xi^1_0 &= \frac{M_0}{M_0+M_1+M_2} = \frac{M_0}{M_0+M_2} \\
\xi^1_1 &= \frac{M_1}{M_0+M_1+M_2}  = 0\\
\xi^1_2 &= \frac{M_2}{M_0+M_1+M_2} = \frac{M_0}{M_0+M_2}
\end{aligned}
$$

<img src="../assets/images/interploation/bc3.png" alt="refine" style="zoom:50%;" />
即

$$
\begin{aligned}
 \xi^2_0 &= \frac{M_0}{M_0+M_1+M_2} = 0 \\
\xi^2_1 &= \frac{M_1}{M_0+M_1+M_2}  = \frac{M_0}{M_1+M_2}\\
\xi^2_2 &= \frac{M_2}{M_0+M_1+M_2} = \frac{M_0}{M_1+M_2}
\end{aligned}
$$

结合上面各式可以得到

$$
\begin{aligned}
\bf y &= \lambda_0 \bf y_0 + \lambda_1 \bf y_1 + \lambda_2 \bf y_2 \\
      &= \lambda_0 \left(\xi^0_0 \bf x_0 + \xi^0_1 \bf x_1 + \xi^0_2 \bf
      x_2\right) + \lambda_1 \left(\xi^1_0 \bf x_0 + \xi^1_1 \bf x_1 + \xi^1_2 \bf
      x_2\right) + \lambda_2 \left(\xi^2_0 \bf x_0 + \xi^2_1 \bf x_1 + \xi^2_2 \bf
      x_2\right)\\
      &= \left(\lambda_0 \xi^0_0 + \lambda_1 \xi^1_0 + \lambda_2 \xi^2_0 \right)
      \bf x_0 + 
      \left(\lambda_0 \xi^0_1 + \lambda_1 \xi^1_0 + \lambda_2 \xi^2_1 \right)
      \bf x_1 +
      \left(\lambda_0 \xi^0_2 + \lambda_1 \xi^1_2 + \lambda_2 \xi^2_2 \right)
      \bf x_2\\
      &= \left(\lambda_0 \xi^0_0 + \lambda_1 \xi^1_0 \right) \bf x_0 + 
      \left(\lambda_0 \xi^0_1 + \lambda_2 \xi^2_1 \right) \bf x_1 +
      \left(\lambda_1 \xi^1_2 + \lambda_2 \xi^2_2 \right) \bf x_2
\end{aligned}
$$

已知三角形ABC的顶点分别为`A(a_0, a_1), B(b_0, b_1), C(c_1, c_2)`,
则该三角形的面积为 `\dfrac{|a_1b_2+b_1c_2+c_1a_2-a_1c_2-c_1b_2-b_1a_2|}{2}`

故加密后三角形内任意一点在大三角形中的重心坐标都可以计算.

最后，当已知某个点在一个单元的重心坐标时，我们就可以计算出这个点在这个单元中的值。
