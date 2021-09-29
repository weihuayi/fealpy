---
title: ADDG求解二阶椭圆问题
permalink: /docs/zh/achievement/ADDG
key: docs-fealpy-ADDG-zh
---


# DDG离散

令 $\Omega \in \mathbb{R}^2$ 是有界多边形开区域, $\partial \Omega = \Gamma_D\cup \Gamma_N$ 且 $\Gamma_D\cap \Gamma_N = \emptyset$, $mes(\Gamma_{D})\ge 0$. 考虑如下二阶椭圆型方程偏微分方程:
$$
-\nabla \cdot (a(x)\nabla u) = f, \qquad & \mathbf{x} \in \Omega, \\
u = g_D, \qquad & \mathbf{x} \in \Gamma_D,\\
\mathbf{n}\cdot(a(x)\nabla u)= g_N, \qquad & \mathbf{x} \in \Gamma_N.
$$
其中, $f \in L^2(\Omega),\,g_D\in H^{\frac{1}{2}}(\Gamma_D),\,g_N\in L^2(\Gamma_N)$ 是已知函数, $\mathbf{n}=(n_1,n_2)$ 是边上的单位外法向量, 扩散系数 $a(x)$ 在 $\Omega$ 中是正的分片常数, 即
$$
a(x) = a_i>0,\,\, x \in \Omega_i,\quad i =1,\cdots,n.
$$
其中开多边形区域 $\{\Omega_i\}_{i = 1}^n$ 是 $\Omega$ 的子域.
## 有限元空间
令 $\mathcal{T}_h$ 是 $\Omega$ 上的三角网格剖分, 假设
(1) $\mathcal{T}_h$ 中的单元是形状正则的, 即:
$$
\forall K\in \mathcal{T}_h,\,\exists c_0>0, \quad s.t \quad h_K/\rho_K \le c_0.
$$
其中 $h_K,\,\rho_K$ 分别是单元 $K$ 的网格尺寸和内接圆直径.
(2) $\mathcal{T}_h$ 是局部拟一致的, 即:
$$
\text{如果} \quad \partial \overline{K_i} \cap \partial \overline{K_j} \ne \varnothing, \,\,\text{则} \quad h_{K_i}\approx h_{K_j}.
$$
(3) 网格 $\mathcal{T}_h$ 是匹配的, 即:
界面 $I = \{\partial \Omega_i \cap \partial \Omega_j:\,i,j = 1,2,\cdots n\}$ 不出穿过 $\mathcal{T}_h$ 中任意单元 $K$.
在剖分 $\mathcal{T}_h$ 上定义间断有限元空间 $V_{DG}^{\ell}$ 和连续有限元空间 $V_h^{\ell}$:
$$
V_{DG}^{\ell} = \{v\in L^2(\Omega):v|_K\in \mathbb{P}_{\ell}(K),\, \forall \,K\in \mathcal{T}_h\},\\
 V_h^{\ell} = \{v\in C(\overline{\Omega}):v|_K\in \mathbb{P}_{\ell}(K),\, \forall \,K\in \mathcal{T}_h\}.
$$

## 跳量和均值
令 $\mathcal{T}_h$ 上边的集合为 $\mathcal{E}$, $K \in \mathcal{T}_h$ 上边的集合为 $\mathcal{E}_K$, 对 $e \in \mathcal{E}$, $h_e$ 为其边长. 定义 $\Omega,\,\Gamma_D,\,\Gamma_N$ 上边的集合分别为:
$$
\mathcal{E}_{I}&=\{e \in \mathcal{E} : e \subset \Omega\}, \\
\mathcal{E}_{D}&=\left\{e \in \mathcal{E} : e \subset \Gamma_{D}\right\}, \\
\mathcal{E}_{N}&=\left\{e \in \mathcal{E} : e \subset \Gamma_{N}\right\}.
$$
为了方便, 记 $\mathcal{E}_{I D}=\mathcal{E}_{I} \cup \mathcal{E}_{D},\,\mathcal{E}_{B}=\mathcal{E}_{D} \cup \mathcal{E}_{N}$.
当 $e \in \mathcal{E}_{I}$, 记 $K_+,\,K_-$ 为 $e$ 的左右单元, 其中 $K_+$ 是整体单元编号较大的单元, $\mathbf{n}$ 为 $e$ 由 $K_+$ 指向 $K_-$ 的单位外法向量. 当 $e \in \mathcal{E}_{B}$, $K_+$ 是 $e$ 上的单元, $\mathbf{n}$ 是 $e$ 的单位外法向量.
对 $v \in V_{DG}^{\ell}$, 跳量的定义为:
$$
[v] := \begin{cases}
            v_+ - v_-,\quad &e \in \mathcal{E}_{I},  \\
            v_+,\quad &e \in \mathcal{E}_{B}.
            \end{cases}
$$
均值的定义为:
$$
\{v\}_w := \begin{cases}
 w_+v_+ + w_-v_-,\quad &e \in \mathcal{E}_{I},  \\
v_+,\quad &e \in \mathcal{E}_{B},
\end{cases}\qquad\qquad\qquad\qquad
\{v\}^w := \begin{cases}
            w_-v_+ + w_+v_-,\quad &e \in \mathcal{E}_I,  \\
            0,\quad &e \in \mathcal{E}_B,
            \end{cases}
$$
其中权重 $w_+,\,w_- \in [0,1]$, 满足: $w_+ + w_- = 1$.
根据以上定义, 经过简单计算可得:
$$
[uv] = \{u\}^w[v] + \{v\}_w[u].
$$
对任意 $e \in \mathcal{E}_{I},\,e = \partial K_+ \cap \partial K_-$, 令 $a_+,\,a_-$ 分别是 $K_+,\,K_+$ 上的扩散系数, 定义 $a$ 在 $e$ 上的加权平均为
$$
W_{e,i} = w_{+,i}\,a_+ + w_{-,i}\,a_-,\quad i = 1,2,3.
$$
$W_{e,i}$ 分别为算数、调和、几何平均
$$
W_{e,1} = \frac{a_+ + a_-}{2},\quad W_{e,2} = \frac{2a_+  a_-}{a_+ + a_-},\quad W_{e,3} = \sqrt{a_+  a_-}, 
$$
对任意 $e \in \mathcal{E}_{B},\,e = \partial K_+\cap\partial \Omega$, $a$ 在 $e$ 上的加权平均为
$$
W_{e,i} = a_+.
$$
为了叙述方便, 以下我们选 $W_e = \{a\}_w =  W_{e,2}$.
## DDG 离散格式
对模型问题第一式两端同乘任意光滑函数 $v$, 在单元 $K$ 上积分, 利用分部积分公式可得:
$$
\int_Ka\nabla u \cdot \nabla v\,\mathrm{dx} - \int_{\partial K}a\nabla u \cdot \mathbf{n} \,v\,\mathrm{ds} = \int_Kfv \,\mathrm{dx}.
$$
记 $u_\mathbf{n} = \nabla u \cdot \mathbf{n},\,  u_{\mathbf{n}\mathbf{n}} = \nabla(u_\mathbf{n})\cdot \mathbf{n}$. 令 $u \in V^{1+ \epsilon}(\mathcal{T}_h)$, 对 $\forall v \in V^{1+ \epsilon}(\mathcal{T}_h)$, 对扩散项使用重复的分部积分得弱形式
$$
\int_K a\nabla u \cdot \nabla v\,\mathrm{dx}+ \int_{\partial K}(\widehat{u}-u)\,a v_\mathbf{n}\,\mathrm{ds} - \int_{\partial K}\widehat {a u_\mathbf{n}}\,\,v\,\mathrm{ds} = \int_Kfv \,\mathrm{dx}.
$$
对所有 $K \in \mathcal{T}_h$ 进行求和, 得
$$
\sum_{K \in \mathcal{T}_h}\int_Kfv \,\mathrm{dx} = \sum_{K \in \mathcal{T}_h}\int_K a\nabla u \cdot \nabla v\,\mathrm{dx}
+ \sum_{K \in \mathcal{T}_h}\int_{\partial K}(\widehat{u}-u)\,a v_\mathbf{n}\,\mathrm{ds} - \sum_{K \in \mathcal{T}_h}\int_{\partial K}\widehat {a u_\mathbf{n}}\,\,v\,\mathrm{ds}.
$$
由跳量的定义和边界条件得:
$$
\sum_{K \in \mathcal{T}_h}\int_{\partial K}(\widehat{u}-u)\,a v_\mathbf{n}\,\mathrm{ds} = \sum_{e \in \mathcal{E}_I}\int_{e} \widehat {u}\,[av_n]-[u\,a v_\mathbf{n}]\,\mathrm{ds} 
 + \sum_{e \in \mathcal{E}_D}\int_{e} (\widehat {u}-u)\,a v_\mathbf{n}\,\mathrm{ds}+ \sum_{e \in \mathcal{E}_N} \int_{e}(\widehat {u}-u)\,a v_\mathbf{n}\,\mathrm{ds},
$$
$$
\sum_{K \in \mathcal{T}_h}\int_{\partial K} \widehat {a u_\mathbf{n}}\,v\,\mathrm{ds} = \sum_{e \in \mathcal{E}_I}\int_{e} \widehat {a u_\mathbf{n}}\,[v]\,\mathrm{ds} + \sum_{e \in \mathcal{E}_D}\int_{e} \widehat {a u_\mathbf{n}}\,v\,\mathrm{ds} + \sum_{e \in \mathcal{E}_N}\int_{e}\widehat {a u_\mathbf{n}}\,v\,\mathrm{ds}.
$$
故
$$
\sum_{K \in \mathcal{T}_h}\int_Kfv \,\mathrm{dx} &= \sum_{K \in \mathcal{T}_h}\int_K a\nabla u \cdot \nabla v\,\mathrm{dx}+ \sum_{e \in \mathcal{E}_I}\int_{e} \widehat {u}\,[av_n]-[u\,a v_\mathbf{n}]\,\mathrm{ds}+ \sum_{e \in \mathcal{E}_N} \int_{e}(\widehat {u}-u)\,a v_\mathbf{n}\,\mathrm{ds}\notag\\
&- \sum_{e \in \mathcal{E}_I}\int_{e} \widehat {a u_\mathbf{n}}\,[v]\,\mathrm{ds} - \sum_{e \in \mathcal{E}_D}\int_{e} \widehat {a u_\mathbf{n}}\,v\,\mathrm{ds} - \sum_{e \in \mathcal{E}_N}\int_{e}\widehat {a u_\mathbf{n}}\,v\,\mathrm{ds}.
$$
定义数值通量:
$$
\widehat{a u_\mathbf{n}}|_e =
\begin{cases}
-\beta_1 h_e^{-1}W_e\,[u] + \{a\,u_\mathbf{n}\}_w - \beta_2h_eW_e\,[u_{\mathbf{n}\mathbf{n}}],\quad e \in \mathcal{E}_{I},\\
-\beta_1 h_e^{-1}W_e\,(u-g_D) + a\,u_\mathbf{n}, \qquad \qquad\qquad e \in \mathcal{E}_{D},\\
g_N, \qquad \qquad\qquad\qquad \qquad\qquad\qquad e \in \mathcal{E}_{N},
\end{cases}
$$
$$
\widehat {u}|_e =
\begin{cases}
\{u\}^w,\qquad& e \in \mathcal{E}_{I},\\
g_D,\qquad& e \in \mathcal{E}_{D},\\
u,\qquad& e \in \mathcal{E}_{N}.
\end{cases}
$$
记单元 $K$ 和 边 $e$ 上的 $L^2$ 内积分别为 $(\cdot,\cdot)_K,\,\langle \cdot,\cdot \rangle_e$. 定义
$$
a_h(u,v)  = \sum_{K \in \mathcal{T}_h}(a\nabla u,\nabla v)_K + \sum_{e \in \mathcal{E}_{ID}}\Big[\beta_1 h_e^{-1}W_e \langle [u],[v]\rangle_e-\langle \{a\,u_\mathbf{n}\}_w,[v]\rangle_e - \langle \{a\,v_\mathbf{n}\}_w,[u]\rangle_e\Big]+\beta_2h_eW_e\sum_{e \in \mathcal{E}_{I} }\langle[u_{\mathbf{n}\mathbf{n}}],[v]\rangle_e,\\
f_h(v) = \sum_{K \in \mathcal{T}_h}(f,v)_K + \sum_{e \in \mathcal{E}_{N}}\langle g_N,v\rangle_e- \sum_{e \in \mathcal{E}_{D}} \langle g_D,a\,v_\mathbf{n}\rangle_e + \beta_1  h_e^{-1}W_e\sum_{e \in \mathcal{E}_{D}} \langle g_D,v\rangle_e .
$$
则相应的 DDG 离散格式为: 求 $u_h\in V_{DG}^{\ell}$, 使得
$$
a_h(u_h,v_h) = f_h(v_h) ,\quad \forall v_h \in V_{DG}^{\ell}.
$$
令 $u$ 是真解, 根据正则性有 $u \in H^{1 + \alpha},\,\alpha>0$. 由于 $f \in L^2(\Omega)$, 故对任意 $0<\epsilon<\alpha$, 容易得到 $u\in V^{1+ \epsilon}(\mathcal{T}_h)$, 因此  $u\in H^{1+\alpha}(\Omega)\cap V^{1+ \epsilon}(\mathcal{T}_h)$.
令 $u \in C^2(\Omega)$, 则有
$$
a_h(u,v) = f_h(v) ,\quad \forall v \in V^{1+ \epsilon}(\mathcal{T}_h).
$$
误差方程:
$$
a_h(u-u_h,v) = 0,\quad \forall v \in  V_{DG}^\ell.
$$
定义 $V_{DG}^{\ell}$ 空间上的能量范数:
$$
||v||_{DG} := \Big(||a^{1/2}\nabla_h v||_{0,\Omega}^2 + \sum_{e \in \mathcal{E}_{ID}}h_e^{-1}W_e||[v]||_{0,e}^2\Big)^{\frac{1}{2}},\qquad v\in V_{DG}^{\ell}.
$$
# 先验误差估计
**定理1[连续性与强制性]**
对于 DDG 方法的双线性型, 存在正的常数 $C_s$ 和 $C_b$, 使得
$$
|a_h(u,v)| &\le C_b ||u||_{DG}||v||_{DG},\quad &\forall u, v \in V_{DG}^{\ell},\\
a_h(u,u) &\ge C_s ||u||_{DG}^2,\quad &\forall u \in V_{DG}^{\ell}.
$$
**定理2[先验误差估计]**
令真解 $u \in C^0(\Omega)$, 离散问题的解 $u_h \in V_{DG}^{\ell}$, 假设在每个子域 $\Omega_j,\,j = 1,\cdots,n$ 中, $u \in H^{\ell +1}(\Omega_j)$ 则有
$$
||u-u_h||_{DG}\le C\left(\sum_{K \in \mathcal{T}_h}h^{2\ell}_K|a^{1/2}u|_{\ell+1, K}^2\right)^{1/2}.
$$
# 后验误差估计
构造残量型后验误差估计：
$$
\eta_R = \left(\sum_{K \in \mathcal{T}_h}\eta_{R,K}^2\right)^{\frac{1}{2}},
$$
在 $K\in\mathcal{T}_h$ 上的局部误差估计定义为：
$$
\eta_{R,K} = (\eta_{R_f,K}^2 +\eta_{J_\sigma,K}^2 +\eta_{J_u,K}^2 +\eta_{R_D,K}^2+ \eta_{R_N,K}^2)^{\frac{1}{2}},
$$
其中
$$
\eta_{R_f,K}^2 &=\frac{h_K^2||f+ \nabla \cdot (a\nabla u_h)||_{0,K}^2}{a_K},\\
\eta_{R_N,K}^2 &= \sum_{e \in \mathcal{E}_K \cap \mathcal{E}_N}\frac{h_e}{a_e}||g_N - a\nabla u_h \cdot \mathbf{n}||_{0,e}^2,\\
\eta_{R_D,K}^2 &= \sum_{e \in \mathcal{E}_K \cap \mathcal{E}_D}\frac{W_e}{h_e}||g_D - u_h||_{0,e}^2,\\
\eta_{J_\sigma,K}^2 &= \frac{1}{2}\sum_{e \in \mathcal{E}_K \cap \mathcal{E}_I}\frac{h_e||[a\nabla u_h \cdot \mathbf{n]}||_{0,e}^2}{W_{e,1}},\\
\eta_{J_u,K}^2& = \sum_{e \in \mathcal{E}_K \cap \mathcal{E}_I}\frac{W_e}{h_e}||[u_h]||_{0,e}^2.
$$
**定理3[可靠性]**
假设 $u \in V^{1+ \epsilon}(\mathcal{T}_h)$ 是变分问题的解, $u_h$ 是数值解, 则基于残量的后验误差估计子 $\eta_R $ 满足如下的全局可靠性
$$
||u-u_h||_{DG}\le C\eta_R.
$$
其中 $C$ 是与网格尺寸和 $a_{max}/a_{min}$ 的比值无关的正常数.
**定理4[有效性]**
假设扩散系数是局部拟单调的, 则存在与网格尺寸和$a_{max}/a_{min}$ 的比值无关的正常数$C$, 使得
$$
\eta_{1,K} \le C\left(||e||_{DG,\omega_K} + osc(\omega_K)\right),\quad \forall K \in \mathcal{T}_h,
$$
其中 $w_K$ 是 $\mathcal{T}_h$ 中与 $K$ 有公共边的单元集合.
# 数值案例
当精确解存在时, 记相对误差为:
$$
rel-err: = \frac{||u-u_h||_{DG}}{||a^{1/2}\nabla u||_{0,\Omega}}.
$$
$$
\eta_1 = \left(\sum_{K \in \mathcal{T}_h}\eta_{R_f,K}^2+\eta_{J_u,K}^2 +\eta_{J_\sigma,K}^2  +\eta_{R_D,K}^2+ \eta_{R_N,K}^2\right)^{\frac{1}{2}},\qquad\qquad
\eta_2 = \left(\sum_{K \in \mathcal{T}_h}\frac{h_K^2||f+ \nabla \cdot (a\nabla u_h)||_{0,K}^2}{a_K}\right)^{1/2},\\
\eta_3 = \left(\sum_{K \in \mathcal{T}_h}\sum_{e \in \mathcal{E}_K \cap \mathcal{E}_I}\frac{W_e}{h_e}||[u_h]||_{0,e}^2\right)^{1/2}, \qquad\qquad\qquad\qquad\qquad\qquad
\eta_4 = \frac{1}{2}\left(\sum_{K \in \mathcal{T}_h}\sum_{e \in \mathcal{E}_K \cap \mathcal{E}_I}\frac{h_e||[k\nabla u_h \cdot \mathbf{n]}||_{0,e}^2}{W_{e,1}}\right)^{1/2}.
$$
**例[间断系数问题]**
$$
-\nabla \cdot (a(x)\nabla u) = f , \qquad  & \mathbf{x} \in \Omega, \\
u = g , \qquad &\mathbf{x} \in \partial \Omega.
$$
其中真解为
$$
u(x,y) = r^{\beta}\mu(\theta),\quad r^2 = x^2 + y^2,\quad tan\,\theta = \frac{y}{x},
$$
$\Omega \in R^2,\Omega = [-1,1]\times [-1,1]$ 为一带 Lipschitz 边界 $\partial \Omega$ 的有界区域, 常数 $a_1 = 161.4476387975881,\, a_2 = 1$, 使得扩散系数满足
$$
a(x) =
\begin{cases}
a_1I, &\quad xy>0\\
a_2I, &\quad xy<0
\end{cases}
$$

<img src="../../../assets/images/figure/exam4_mesh1_2.png" alt="workflow" style="zoom:42%;" /><img src="../../../assets/images/figure/exam4_mesh2_2.png" alt="workflow" style="zoom:42%;" /><img src="../../../assets/images/figure/exam4_mesh4_2.png" alt="workflow" style="zoom:42%;" />

<img src="../../../assets/images/figure/exam4_estimator1_2.png" alt="w" style="zoom:32%;" /><img src="../../../assets/images/figure/exam4_estimator2_2.png" alt="w" style="zoom:32%;" /><img src="../../../assets/images/figure/exam4_estimator2_2.png" alt="w" style="zoom:32%;" />







# 参考文献
<div id="refer-anchor-1"></div>
[1] H. Cao, Y. Huang, N. Yi. Adaptive direct discontinuous Galerkin method for elliptic equations. Computers and Mathematics with Applications, 97: 394-415, 2021.(https://doi.org/10.1016/j.camwa.2021.06.014)

