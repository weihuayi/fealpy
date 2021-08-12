<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Fourth-Order-Elliptic 问题

## PDE 模型

考虑四阶问题:

$$
\begin{equation}
\begin{cases}
\Delta^2 u - \Delta u = f & in \quad \Omega\\
u = g_1, \quad \frac{\partial u}{\partial \boldsymbol n} = g_2 & on \quad \partial \Omega
\end{cases}
\end{equation}
$$

令 $$-\Delta u = \sigma$$, 将四阶问题化为二阶问题:

$$
\begin{equation}
\begin{cases}
-\Delta u - \sigma = 0 & in \quad \Omega\\
-\Delta \sigma + \sigma = f & in \quad \Omega\\
u = g_1, \quad \frac{\partial u}{\partial \boldsymbol n} = g_2 & on \quad \partial \Omega
\end{cases}
\end{equation}
$$

## 理论准备
记:

$$
(v_0, v_1)_K = \int_K v_0v_1 \mathrm dx, \quad (v_0, v_1)_{\mathcal T} = \sum_{K\in \mathcal T}\int_K v_0v_1 \mathrm dx,
\quad (v_0, v_1)_{\Gamma} = \sum_{E\in\Gamma}\int_E v_0v_1 \mathrm dx
$$


$$
\{u\}_{E\in\Gamma_I} = \frac{1}{2}(u|_{E^+} + u|_{E^-}), \quad
[u]_{E\in\Gamma_I} = u|_{E^+} - u|_{E^-},
$$


$$
\{u\}_{E\in\Gamma_B} = u, \quad
[u]_{E\in\Gamma_B} = u
$$

我们有:

$$
[uv]_{E\in\Gamma_I} = \{u\}_E[v]_E + [u]_E\{v\}_E, \quad
[uv]_{E\in\Gamma_B} = uv
$$

记空间:

$$
H^1(\mathcal T) = \{v| \forall K \in \mathcal T, v|_K \in H^1(K)\}
$$

使用分部积分, 我们有 $$\forall v_0, v_1 \in H^1(\mathcal T)$$:

$$
\begin{align}
(\nabla v_0, \nabla v_1)_{\mathcal T} = -(\Delta v_0, v_1)_{\mathcal T} +
(\left[\frac{\partial v_0}{\partial \boldsymbol n}\right], \{ v_1\} )_{\Gamma_I}+
(\left\{\frac{\partial v_0}{\partial \boldsymbol n}\right\}, [v_1] )_{\Gamma}.
\end{align}
$$

## 弱形式
由 $$(3)$$ 可知:  $$\forall v\in H^1(\mathcal T)$$:

$$
\begin{align}
(\nabla u, \nabla v)_{\mathcal T} = -(\Delta u, v)_{\mathcal T} +
(\left[\frac{\partial u}{\partial \boldsymbol n}\right], \{ v\} )_{\Gamma}+
(\left\{\frac{\partial u}{\partial \boldsymbol n}\right\}, [v] )_{\Gamma_I}.
\end{align}
$$

因为: $$[u]_{\Gamma_I} = 0, \left[\frac{\partial u}{\partial \boldsymbol n}\right]_{\Gamma_I} = 0, 
\left\{\frac{\partial u}{\partial \boldsymbol n}\right\}_{\Gamma_B} = g_2.$$ 所以有:
$$
\begin{align*}
(\nabla u, \nabla v)_{\mathcal T} - (\sigma, v)_{\mathcal T} -
(\left\{\frac{\partial u}{\partial \boldsymbol n}\right\}, [v] )_{\Gamma_I} -
(\left\{\frac{\partial v}{\partial \boldsymbol n}\right\}, [u])_{\Gamma}
= (g_2, v)_{\Gamma_B} - (\frac{\partial v}{\partial \boldsymbol n}, g_1)_{\Gamma_B}
\end{align*}
$$

同理我们有:

$$
\begin{align}
(\nabla v, \nabla \sigma)_{\mathcal T} - (\sigma, v)_{\mathcal T} -
(\left\{\frac{\partial \sigma}{\partial \boldsymbol n}\right\}, [v] )_{\Gamma} -
(\left\{\frac{\partial v}{\partial \boldsymbol n}\right\}, [\sigma])_{\Gamma_I}
=  (f, v)_{\mathcal T}
\end{align}
$$

记双线性型:

$$
J(u, v) = ([u], [v])_{\Gamma}, \quad J_B(u, v) = ([u], [v])_{\Gamma_B}, \quad J_I(u, v) = ([u], [v])_{\Gamma_I}, \quad
Q_I(u, v) = (\left[ \frac{\partial u}{\partial \boldsymbol n}\right], \left[ \frac{\partial v}{\partial \boldsymbol n}\right])_{\Gamma_I}
$$

$$
A(u, v)  =(\nabla u, \nabla v)_{\mathcal T} - (\left\{\frac{\partial u}{\partial \boldsymbol n}\right\}, [v] )_{\Gamma_I} -
(\left\{\frac{\partial v}{\partial \boldsymbol n}\right\}, [u])_{\Gamma} + J_I(u, v) + Q_I(u, v)
$$

记 $$H^1(\Omega)$$ 上的线性泛函:

$$
\begin{align*}
F_1(u) = (g_2, v)_{\Gamma_B} - (\frac{\partial v}{\partial \boldsymbol n}, g_1)_{\Gamma_B}, \quad F_2(u) = J_B(g_1, v) + (f, v)_{\mathcal T}
\end{align*}
$$

因为 $$J(u, v) = J_B(g_1, v), \quad Q_I(u, v) = J_I(u, v) = 0$$, 由 $$(4), (5)$$ 可得 $$(2)$$ 的弱形式:

$$
\begin{equation*}
\begin{cases}
A(u, v) - (\sigma, v) = F_1(u)\\
A(v, \sigma) + (\sigma, v) + J(u, v) = F_2(u)
\end{cases}
\end{equation*}\quad \forall v \in H^1(\mathcal T)
$$















