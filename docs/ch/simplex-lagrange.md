# 单纯形网格上的 Lagrange 有限元 

这里以 Poisson 方程为例介绍单纯形网格上的任意次 Lagrange 有限元方法。

给定区域 $\Omega\subset\mathbb R^d$, 其边界 $\partial \Omega = \Gamma_D \cup \Gamma_N
\cup \Gamma_R$

$$
-\Delta u = f, \quad\text{in }\Omega
$$

满足如下的边界条件：

$$
u = g_D, \quad\text{on }\Gamma_D\leftarrow \text{\bf Dirichlet } 
$$

$$
\frac{\partial u}{\partial\boldsymbol n}  = g_N, \quad\text{on }\Gamma_N \leftarrow \text{\bf Neumann}
$$

$$
\frac{\partial u}{\partial\boldsymbol n} + \kappa u = g_R, \quad\text{on }\Gamma_R \leftarrow \text{\bf Robin}
$$


在 Poisson 方程两端分别乘以测试函数 $v \in H_{D,0}^1(\Omega)$, 利用分部积分，可得到其对应的**连续弱形式**

$$
(\nabla u,\nabla v)+<\kappa u,v>_{\Gamma_R} = (f,v)+<g_R,v>_{\Gamma_R}+<g_N,v>_{\Gamma_N}
$$

进一步，取一个 $N$ 维的有限维空间 $V_h = \operatorname{span}\{\phi_i\}_0^{N-1}$，其基函数向量记为

$$
\bm\phi = [\phi_0, \phi_1, \cdots, \phi_{N-1}],
$$

注意这 $\bm\phi$ 是行向量函数，则 $V_h$ 中任何一个函数 $u_h$ 都对应一个唯一的 $N$ 维列向量 $\boldsymbol u$, 满足 

$$
u_h = \bm\phi \boldsymbol u=\bm\phi
\begin{bmatrix}
u_0\\ u_,\\ \vdots\\ u_{N-1}
\end{bmatrix}
$$

如果用 $V_h$ 替代无限维的空间 $H^1_{D,0}(\Omega)$, 从而得到原问题的**离散弱形式**：求 $u_h\in V_h$, 满足

$$
\left(\nabla u_h, (\nabla \bm\phi)^T\right)+<\kappa u_h, \bm\phi^T>_{\Gamma_R}=
(f, \bm\phi^T)+<g_R, \bm\phi^T>_{\Gamma_R}+<g_N, \bm\phi^T>_{\Gamma_N},
$$

最终转化为如下离散代数系统

$$
(\boldsymbol A + \boldsymbol R)\boldsymbol u = \boldsymbol b + \boldsymbol b_N+ \boldsymbol b_R
$$

其中

$$
\boldsymbol A = \int_\Omega (\nabla \bm\phi)^T \nabla\bm\phi\mathrm d\boldsymbol x
\quad \boldsymbol R = \int_{\Gamma_R} \bm\phi^T \bm\phi\mathrm d \boldsymbol s
$$

$$
\boldsymbol b = \int_\Omega f\bm\phi^T\mathrm d \boldsymbol x,\quad
\boldsymbol b_N =  \int_{\Gamma_N} g_N\bm\phi^T\mathrm d \boldsymbol s,\quad
\boldsymbol b_R =  \int_{\Gamma_R} g_R\bm\phi^T\mathrm d \boldsymbol s
$$

## 重心坐标与几何单纯形网格

记 $\{\boldsymbol x_i:=[x_{i,0}, x_{i, 1}, \ldots, x_{i, d-1}]\}_{i=0}^d$ 为 $\mathbb R^d$ 空
间中的一组点, 假设它们不在同一个超平面上, 也即是说 $d$ 个向量 $\boldsymbol x_0\boldsymbol x_1$,
$\boldsymbol x_0\boldsymbol x_2$, $\cdots$, 和 $\boldsymbol x_0\boldsymbol x_d$ 是线性无关的, 等价于矩阵

$$
    \boldsymbol A =\begin{bmatrix}
        x_{0, 0} & x_{1, 0} & \cdots & x_{d, 0} \\
        x_{0, 1} & x_{1, 1} & \cdots & x_{d, 1} \\
        \vdots   & \vdots   & \ddots & \vdots \\
        x_{0, d-1} & x_{1, d-1} & \cdots & x_{d, d-1}\\
        1 & 1 & \cdots & 1
    \end{bmatrix}
$$

是非奇异的.
## Lagrnage 有限元方法 

```python
from fealpy.decorator import cartesian
class CosCosData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return ( y == 1.0) | ( y == 0.0)
```
