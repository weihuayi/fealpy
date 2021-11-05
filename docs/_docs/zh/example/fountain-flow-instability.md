---
title: 流动前沿不稳定模拟
permalink: /docs/zh/example/fountain-flow-instability
key: docs-fountain-flow-instability-zh
author: wpx
---

# 模型

## 几何区域

最初，通道部分地填充有矩形聚合物块，其被活塞移动到通道的空部分中

<div align="center">
    	<img src='../../../assets/images/example/fountain-flow/geometry-0.png' width="500"> 
</div>

为了避免建模运动流域的问题，将活塞速度从速度场中减去

<div align="center">
    	<img src='../../../assets/images/example/fountain-flow/geometry-1.png' width="500"> 
</div>

给不同区域相应的符号，注意，由于活塞和墙壁之间的速度跳跃，在活塞的拐角处的流场中引入了两个奇点。

<div align="center">
    	<img src='../../../assets/images/example/fountain-flow/geometry.png' width="500"> 
</div>

- $\Omega_m$ ：融熔物区域，建模为恒温不可压缩粘弹性液体，且由于粘性力占主导，惯性和引力忽略，
- $\Omega_g$ ：气体区域，建模为可压缩低粘性流体
- $\Gamma_p$ ：活塞边界
- $\Gamma_{wp}$ ：融熔物区域的移动壁
- $\Gamma_{wg}$ ：气体区域的移动壁
- $\Gamma_{gin}$ ：气体区域流入边界
- $\Gamma_f$ ：流动前沿



## 控制方程

$$
\begin{aligned}
- \nabla \cdot \boldsymbol \sigma &= 0 \quad \text{in} \quad \Omega \\
\nabla \cdot \boldsymbol u &= 0  \quad \text{in} \quad \Omega_m
\end{aligned}
$$

其中 $\boldsymbol u$ 为速度向量，$\Omega = \Omega_m \cup \Omega_g$，  $\boldsymbol \sigma$ 为柯西应力张量，其气体域和融熔域表示分别为

$$
\begin{aligned}
\boldsymbol \sigma = -p \boldsymbol I + \boldsymbol \tau \quad \text{in} \quad \tau_m \\
\begin{cases}
\boldsymbol \sigma = -p \boldsymbol I + 2 \eta_g \boldsymbol D \\
 \eta_g \nabla \cdot \boldsymbol u + p =0   \\
\end{cases}
\quad \text{in} \quad \tau_g
\end{aligned}
$$

其中 $p$ 是压力， $\boldsymbol D$ 是形变率张量， $\boldsymbol \tau$ 是粘弹性外应力张量，其选用XPP模型来刻画

$$
\begin{aligned}
	\boldsymbol \tau  = G(\boldsymbol c - \boldsymbol I)\\
	\frac{\partial \boldsymbol c}{\partial t} + \boldsymbol u \cdot \nabla \boldsymbol c - (\nabla \boldsymbol u)^T \cdot \boldsymbol c
	- \boldsymbol c \cdot \nabla \boldsymbol u + \boldsymbol f_{rel}(\boldsymbol c) = 0 \\ 
		\boldsymbol f_{rel} = 2\frac{\exp(v(\sqrt{tr \boldsymbol c /3}-1))}{\lambda_s}(1-\frac{3}{tr \boldsymbol c})\boldsymbol c \\
		+ \frac{1}{\lambda_b}(\alpha \boldsymbol c \cdot \boldsymbol c + \frac{3}{tr \boldsymbol c}
		[1-\alpha -\frac{\alpha}{3}tr(\boldsymbol c \cdot \boldsymbol c)]\boldsymbol c+(\alpha-1)\boldsymbol I)
\end{aligned}
$$

参数说明：

- $\boldsymbol c$ : 构形张量
- $G$ ：弹性模量
- $ \boldsymbol f_{rel} : $是一个非线性松弛张量
- $q$ :悬臂的数量
- $v=\frac{2}{q}$ 
- $\lambda_s$ :方向松弛时间
- $\lambda_b$ :主干松弛时间
- $\alpha$ :各项异性的滑移参数，设置为0
- $r = \frac{\lambda_b}{\lambda_s}$ :用来衡量管段或者纠缠物的数量



## 流动前沿刻画

