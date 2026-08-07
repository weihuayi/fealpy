# NS 形状导数说明

本文档说明弯管 `NS` 形状优化中，论文中的严格形状导数应该是什么样子，以及它和当前 `cashocs_implement` 代码链路的对应关系。

参考论文：

- `C:\Users\A208-29\Desktop\006888454.pdf`
- Schmidt, S. and Schulz, V., *Shape Derivatives for General Objective Functions and the Incompressible Navier--Stokes Equations*

这份 PDF 是 2009 年预印本，正式发表版是 2010 年 *Control and Cybernetics* 的文章。两者的核心结论一致：**最终形状导数必须写成 Hadamard 形式，只依赖边界法向扰动 `V·n`。**

---

## 1. 论文里到底在证明什么

论文考虑的目标函数是

```math
J(u,p,\Omega)
= \int_\Omega f(u,Du,p)\,dA
  + \int_{\Gamma_0} g(u,D_nu,p,n)\,dS.
```

其中：

- `\Omega` 是流体区域；
- `\Gamma_0` 是设计边界；
- `u`、`p` 分别是速度和压力；
- `f` 是体积分目标；
- `g` 是边界积分目标。

论文的核心结论是：经过状态方程、伴随方程和形状微分消元以后，最后必须得到

```math
dJ(\Omega)[V] = \int_{\Gamma_0} G_{\mathrm{NS}}(u,p,\lambda)\,(V\cdot n)\,dS.
```

这里：

- `G_NS` 是一个**标量边界密度**；
- `V` 是边界扰动场；
- `n` 是外法向；
- 真正进入优化的只应该是 `V·n`。

这就是 Hadamard 结构。它告诉我们：

```math
\text{shape derivative} \;\neq\; \text{任意 traction 向量},
```

而是必须先整理成一个标量密度，再乘上法向位移。

---

## 2. 论文的通用 Hadamard 结果怎么理解

论文第 4 节给出了 NS 的一般 Hadamard 形式。把它写成更容易读的样子，可以概括为：

```math
dJ(u,p,\Omega)[V]
= \int_{\Gamma_0} (V\cdot n)\; G_{\mathrm{NS}}(u,p,\lambda)\,dS.
```

其中 `G_NS` 不是单一一项，而是由下面几部分组成：

```math
G_{\mathrm{NS}}
=
f(u,Du,p)
+ \text{(来自边界项 } g \text{ 的贡献)}
+ \text{(来自伴随 } \lambda \text{ 的贡献)}
+ \text{(来自法向与曲率的几何项)}.
```

更具体地说，论文的 Lemma 4.6 把边界密度分成四块：

```math
G_{\mathrm{NS}}
=
f(u,Du,p)
+ \Bigl[\langle \nabla g, n\rangle + \kappa g\Bigr]
+ \Bigl[-\partial_u g + \mu\,\partial_n \lambda + \sum_{i,j}\frac{\partial f}{\partial a_{ij}}\partial_n u_i \Bigr]
+ \Bigl[\operatorname{div}_\Gamma \nabla_n g - \kappa \langle \nabla_n g,n\rangle\Bigr].
```

说明：

- `\kappa` 是曲率项；
- `a_{ij}` 表示 `Du` 的分量；
- `\partial_n u` 表示法向导数；
- `\partial_n \lambda` 表示伴随的法向导数。

这段公式的重点不是每一项的记号细节，而是结构：

1. 先有体积分 `f`；
2. 再有边界积分 `g`；
3. 再由伴随消掉局部形状导数；
4. 最后剩下一个**纯边界标量密度**。

这就是我们需要的严格版本。

---

## 3. dissipation 示例为什么特别重要

论文后面的示例就是我们关心的能量耗散问题。它的体积分目标是

```math
f(u,Du,p) = \mu \sum_{i,j=1}^d \left(\frac{\partial u_i}{\partial x_j}\right)^2,
```

并且边界项取

```math
g = 0.
```

这意味着我们当前关心的就是“耗散”这个最常见的 NS 目标。

论文给出的伴随方程可以概括为

```math
\mu \Delta \lambda_i
- \rho \sum_{j=1}^d
  \left(
    \frac{\partial \lambda_j}{\partial x_i}u_j
    +
    \frac{\partial \lambda_i}{\partial x_j}u_j
  \right)
- \frac{\partial \lambda_p}{\partial x_i}
= -2\mu \Delta u_i
```

并且满足

```math
\operatorname{div}\lambda = 0.
```

对我们当前的弯管耗散实现，更关键的是右端项应该理解为

```math
-2\mu \Delta u,
```

也就是耗散目标对状态变量的严格导数。我们在代码里优先用有限元离散下的
拉普拉斯弱式等价来装配它，而不是继续用“对流项 + 压力梯度”的残差拼法。

边界条件中，和我们最相关的是设计壁面 `\Gamma_0` 上：

```math
\lambda = 0 \quad \text{on } \Gamma_0.
```

论文随后指出：因为 `\Gamma_0` 上的这组条件成立，所以最终梯度可以化简为一个**只依赖边界法向扰动的标量密度**。OCR 里能直接读到的主项就是

```math
\mu \sum_{i,j=1}^d \left(\frac{\partial u_i}{\partial x_j}\right)^2.
```

对这篇文档对应的弯管耗散特例，论文第 5.1 节最后简化后可读出的主项是

```math
G_{\mathrm{paper}}
=
\mu \sum_{i=1}^d \left(\frac{\partial u_i}{\partial n}\right)^2.
```

这里需要特别注意两点：

1. `\lambda` 在论文里用于建立伴随方程，但在这个特例的最终边界密度里并不再显式出现；
2. 也就是说，论文这个弯管耗散例子最后得到的是**只依赖 state 法向导数的 Hadamard 密度**。

如果把它和我们当前代码里的严格接口对照，那么我们现在的默认实现已经回到论文式最简版本，
也就是把最终边界密度默认收缩为

```math
G_{\mathrm{NS}} = \mu\left|\partial_n u\right|^2.
```

同时，代码里仍然保留了一个显式的扩展分支，必要时可以切回到
“保留伴随修正项”的版本，

```math
G_{\mathrm{ext}}
=
\mu\left(\left|\partial_n u\right|^2-\partial_n \lambda\cdot \partial_n u\right),
```

其中

```math
\partial_n u = (\nabla u)n,\qquad \partial_n \lambda = (\nabla \lambda)n.
```

也就是说，实际进入更新的方向是

```math
dJ(\Omega)[V] = \int_{\Gamma_0} G_{\mathrm{NS}}(V\cdot n)\,dS.
```

因此更准确的说法是：

- **论文的弯管耗散例子**最终给出的是 `G_paper = \mu \sum_i (\partial_n u_i)^2`；
- **我们当前代码默认已经对齐到这一版**；
- **扩展分支**才会保留 `\partial_n \lambda` 的修正项，用于以后别的更一般目标。

---

## 4. 为什么 Stokes 路线“看起来还能跑”

Stokes 之所以常常看起来没问题，原因主要有四个：

1. 方程是线性的；
2. 伴随结构接近自伴随；
3. 没有对流项把局部边界应力和全局目标敏感性明显拆开；
4. 代理型 traction 往往和真梯度方向比较接近。

所以在障碍物 Stokes 里，一个 traction-like 的代理量经常还能给出可下降的方向。

但这并不表示它是严格的形状导数，只能说明它“工程上常常够用”。

---

## 5. 为什么 NS 里不能继续只用代理 traction

稳态 NS 和 Stokes 最大的不同是对流项：

```math
\rho (u\cdot\nabla)u.
```

这个项会让形状变化对目标的影响变得更强，也更非局部。  
如果我们只拿边界上的物理牵引

```math
t(u,p)
= \sigma(u,p)n
= \bigl(-pI+\mu(\nabla u+\nabla u^T)\bigr)n
```

来当最终梯度，那么做的事情其实是：

```math
\text{把物理应力} \quad \text{误当成} \quad \text{shape derivative}.
```

这在 NS 下通常不成立。

对你们弯管问题来说，这个失配尤其明显，因为：

- 流动存在惯性；
- 弯管几何会引起强烈的速度重分布；
- 你们日志里已经出现过“预测导数为负，但微扰后目标上升”的现象；
- 这正是代理梯度和真实 Hadamard 密度不一致的典型信号。

---

## 6. 我们真正应该实现的最小公式

对弯管 NS，最小正确目标应该写成：

```math
dJ(\Omega)[V]
= \int_{\Gamma_{\mathrm{design}}}
  G_{\mathrm{NS}}(u,p,\lambda)\,(V\cdot n)\,dS.
```

对当前弯管耗散特例，我们按上面的最小严格公式实现：

```math
G_{\mathrm{NS}}
=
\mu\left(\left|\partial_n u\right|^2-\partial_n \lambda\cdot \partial_n u\right).
```

这意味着优化系统中真正参与更新的量应该是：

```math
t_{\mathrm{shape}} = G_{\mathrm{NS}}\,n.
```

注意这里的 `t_shape` 只是**边界法向代表**，它本身不是 shape derivative；  
真正的 shape derivative 是前面的标量密度 `G_NS`。

所以最小可实现流程应该是：

1. 解 NS state；
2. 解与该 state 一致的 NS adjoint；
3. 在设计边界上计算标量密度 `G_NS`；
4. 把 `G_NS n` 乘上边界顶点的局部测度权重，再送入几何梯度 / Riesz projection；
5. 由几何梯度层完成网格平滑和位移传播。

---

## 7. 现在代码里实际是什么状态

### 7.1 `objective.py`

这里负责构造目标函数和伴随右端：

- `calculate_shape_derivative(...)` 只是一个转发入口；
- 在当前弯管 benchmark 里，真正的 `shape_derivative` 主要由 `adjoint_solver.py` 填充到 `adjoint_result`；
- 所以这里不是“自己发明梯度”，而是把上游已经算好的严格边界密度往下传。

### 7.2 `adjoint_solver.py`

这里才是当前弯管 NS 目标的关键位置。

- 当 `shape_density_variant` 取 `paper`、`paper_dissipation` 或 `paper_dissipation_only` 时，默认构造的是论文式最简密度 `G_NS = μ |∂_n u|^2`；
- 返回值同时给出 `shape_derivative = G_NS n` 和 `shape_density = G_NS`；
- 只有当显式切到非 paper 变体时，才会额外把伴随修正项 `-∂_n λ · ∂_n u` 加回去。

### 7.3 `geometry_gradient.py`

这一层做的是：

- 接收上游传来的导数信息；
- 做投影；
- 做 Riesz 平滑；
- 做网格重新扩展。

它应该是“使用者”，不应该自己发明 shape derivative。

### 7.4 `benchmark_elbow_pipe_stokes.py`

这个文件里保留了 `build_shape_derivative_source(...)`，但在当前 paper 配置下，它不是主导路径；
真正主导的是 `adjoint_solver.py` 里构造出来的 `shape_density_variant="paper"` 分支。

此外还有一个接线问题需要注意：

```python
options["state_system_is_linear"] = False
```

但 `state_solver.py` 实际读取的是 `fluid_model.state_system_is_linear`。  
这意味着如果 fluid model 没有同步设置这个字段，就可能把非线性 NS 分支误当成线性系统。

### 7.5 这次重新核对后的结论

当前这条 paper 弯管链路里，`adjoint_solver.py` 已经在用论文式的 Hadamard 密度做边界形状导数，`geometry_gradient.py` 只是对这个边界代表做投影和传播。
因此“每步下降很小”更像是问题尺度和允许位移都比较保守，不是明显的数学公式对照错误。

---

## 7.6 法向更新口径补充

为了避免“法向更新被后处理搅乱”，这条 NS 路线现在按下面的口径推进：

1. 平滑优先作用在法向标量速度上，不直接对二维位移向量做邻点平均。
2. 面积保持投影也优先在法向标量速度上做，然后再重建成 `u = g n`。
3. 线弹性传播只负责把边界位移扩展到内部，不改边界上的 Dirichlet 位移本身，所以不会主动破坏你已经指定好的法向更新。
4. 设计边界顺序必须是 `boundary_cycle` 的有序子序列，不能再用无序集合差直接拼。
5. 固定节点必须显式清零，不能依赖 `setdefault` 这种“已经存在就不覆盖”的写法。
6. 在 `strict_normal_boundary_update` 打开时，线搜索优先以最终试探目标值做接受判断，避免中间态方向导数和实际位移不一致。
7. `assemble_geometry_gradient()` 最终送给优化器的量必须是真正的下降方向，不能把 raw gradient 直接当成 `descent_direction`。当前实现里，`descent_direction` 应该是 `node_gradient` / `normal_gradient` / `raw_gradient` 里选出来的那一个再取负号。

这次已经开始做的动作是：

- `benchmark_elbow_pipe_stokes.py` 里把 `design_boundary_node_order` 改成了按 `boundary_cycle` 过滤后的有序 `design_nodes`。
- `geometry_gradient.py` 里新增了 `strict_normal_boundary_update` 兼容路径，平滑和面积投影都可以保持法向方向不被打散。

---

## 8. 对照结论

### Stokes

- 可以继续保留现状；
- 代理 traction 在工程上往往够用；
- 但它不是严格 Hadamard 形状导数的最干净实现。

### NS

- 不能继续沿用 Stokes 的 traction 代理；
- 应该回到论文中的 Hadamard 标量密度；
- 最终必须是 `G_NS (V·n)`，而不是直接拿 `t(u,p)` 当导数。

---

## 9. 建议的改造顺序

如果后面要把 NS 真正修正成严格版本，建议按这个顺序：

1. 先把 `G_NS` 的数学表达式确定清楚；
2. 再把 `objective.py` 里的 shape derivative 接口改成返回标量密度；
3. 再让 `adjoint_solver.py` 只负责求解 state / adjoint，不再拼 traction 代理；
4. 最后让 `geometry_gradient.py` 只负责投影和平滑；
5. 用 Taylor test 验证符号和方向。

这样最稳。

---

## 10. 论文里除了 dissipation 之外，还讨论了什么目标

这篇论文并不只讲“能量耗散”这一种目标。它在开头就把目标函数写成更一般的形式：

```math
J(u,p,\Omega)
= \int_\Omega f(u,Du,p)\,dA
  + \int_{\Gamma_0} g(u,D_nu,p,n)\,dS.
```

这意味着它同时覆盖两大类目标：

### 10.1 体积分目标

体积分目标就是定义在整个流体区域 `\Omega` 上的量，比如：

```math
\int_\Omega f(u,Du,p)\,dA.
```

论文明确提到过的例子是：

- 体积耗散 / 能量损失；
- 一般的 volume objective；
- 内流中的压降最小化也可以放进这类框架。

### 10.2 边界积分目标

边界积分目标是定义在设计边界 `\Gamma_0` 上的量，比如：

```math
\int_{\Gamma_0} g(u,D_nu,p,n)\,dS.
```

论文特别点名的例子有：

- `drag`，阻力；
- `lift`，升力；
- 匹配目标表面压力分布；
- 其他依赖法向、曲率或边界几何的表面功能量。

### 10.3 论文里强调的“约束”

如果你说的“约束”是优化问题中的附加约束，那么这篇文章真正的主约束其实是：

```math
\text{不可压 NS 状态方程} + \text{边界条件}.
```

它并没有在主定理里专门加入一个独立的体积约束、重心约束或曲率约束。  
它更强调的是：

1. 目标函数可以很一般；
2. 但边界项 `g` 不能乱写；
3. 因为压力在不可压 NS 里没有独立边界条件，所以 `g` 必须满足某些兼容条件，才能导出一致的伴随边界条件。

论文中给出的典型边界兼容条件是：

```math
\lambda_i = \frac{1}{\mu}\frac{\partial g}{\partial b_i}
\quad\text{on } \Gamma_0,
```

并且压力方向还要满足类似

```math
\langle \lambda, n\rangle = -\frac{\partial g}{\partial p}
\quad\text{on } \Gamma_0.
```

也就是说，论文里的“约束”重点不是额外再加一个几何约束，而是**边界目标必须和 NS 的伴随边界条件兼容**。

### 10.4 对我们代码的含义

这点对我们很重要，因为它说明：

- Stokes 里如果只是做一个 traction-like 代理，通常还能蒙对方向；
- 但 NS 里如果目标包含边界量，必须先保证 `g` 的形式是兼容的；
- 否则伴随边界条件不闭合，最终的形状导数就会失真。

换句话说，论文不是只给了“一个 dissipation 公式”，而是给了一个**能同时处理体目标和边界目标的通用框架**。

如果后面我们要把弯管 NS 做严谨，最重要的不是把某个 traction 再调一调，而是先确认：

```math
我们的目标函数到底是体目标、边界目标，还是两者混合；
```

然后再按论文的 Hadamard 结构去写对应的 `G_NS`。

---

## 11. 论文里的几个弯管算例到底在优化什么

你提到的“几个符合的弯管算例”，对应论文第 5 节里的两个内部流算例：

### 11.1 Flow Through a Pipe

这个算例是一个连接两点的管道优化问题。论文明确写的是：

```math
\text{The objective is to minimize the dissipation of kinetic energy into heat.}
```

对应的目标函数就是

```math
f(u,Du,p)
= \mu \sum_{i,j=1}^d \left(\frac{\partial u_i}{\partial x_j}\right)^2,
 \qquad
g=0.
```

也就是说，它优化的是：

- **体积分耗散**；
- 也可以理解成**压降相关的能量损失最小化**。

这个算例里还加了一个形状约束：

```math
\text{volume must be preserved}.
```

论文里是通过每次形状更新后的投影步骤来保持体积不变。

### 11.2 Flow through a T-Connection

这个算例是一个 T 形连接管，同样是内部流优化。论文同样写得很直接：

```math
\text{The initial junction has a net loss of kinetic energy into heat.}
```

所以这里优化的目标仍然是：

```math
J(u,\Omega) = \int_\Omega \mu \sum_{i,j=1}^d
\left(\frac{\partial u_i}{\partial x_j}\right)^2 dA.
```

并且这个算例还加了一个面积/体积保持条件：

```math
\text{The total area occupied by the fluid was enforced to stay the same}.
```

所以这个例子本质上也是：

- **最小化黏性耗散 / 压降损失**；
- **同时保持总面积不变**。

### 11.3 这和我们弯管 NS 的关系

这说明论文里的这些内部流算例，核心目标不是 drag 或 lift，而是：

```math
\text{minimize viscous dissipation}
\quad \Leftrightarrow \quad
\text{reduce pressure loss along the channel}.
```

这和我们现在弯管 NS benchmark 的物理意义是对得上的。  
区别只在于：

- 论文里已经把这个目标写成严格 Hadamard 形状导数；
- 我们当前代码里还停留在代理 traction 或不完全闭合的实现阶段。

### 11.4 论文图 5.1 的几何怎么落到代码

图 5.1 可以先抽象成一个宽度为 `1.0` 的 2D 管道，中心线由三段直线组成：

```math
(0, 0) \rightarrow (L_{in}, 0) \rightarrow (L_{in} + L_b, H) \rightarrow (L_{in} + L_b + L_{out}, H).
```

其中：

- 左端和右端是平直截面，对应入口和出口；
- 中间两处拐角用圆角或折线采样近似；
- 图中的 `fixed / variable` 可以理解为：两端固定，中间弯折段是设计区。

在代码里，这类几何最适合单独放在一个 2D mesher 里，比如 `PaperPipeMesher2D`，而不是继续复用 90 度弯管的 FSI 网格器。

---

## 12. 论文算例与当前实现的逐项对照

下面这张表把论文的两个内部流算例和我们当前的弯管 benchmark 放在一起对照。

| 项目 | 论文 Flow Through a Pipe / T-Connection | 当前 `benchmark_elbow_pipe_stokes.py` |
| --- | --- | --- |
| 几何 | 2D 内流管道；Pipe 是连接两点的管道；T-Connection 是底部入口、左右出口的 T 形连接 | 90 度弯管，`ElbowPipeMesher` 生成；默认参数 `D=1.0`，`bend_angle=90.0`，`R_bend_inner=2.3`，`L_in_ratio=4.0`，`L_out_ratio=5.0`，`wall_thickness=0.05` |
| 设计边界 | 论文的变量边界是管壁/障碍物边界 | 当前设计边界是弯管壁面 `wall/design` |
| 入口条件 | Pipe 使用抛物线入口，最大速度 `u1=1.5`；T-Connection 为二次型入口 | 当前入口也是抛物线型，代码写成 `u1 = 0.25 * (y-y_min) * (y_max-y)`，`u2=0` |
| 出口条件 | 论文写的是自然出流条件 `pn - \mu \partial_n u = 0` | 当前代码对出口使用压强 Dirichlet `p=0`，并额外加了 `pressure_integral_target=0.0` 的拉格朗日处理 |
| 物性参数 | Pipe 明确给出 `\mu=400`，`\rho=1.0`，`Re=400`；T-Connection 给出 `Re=100` | 当前默认 `\rho=1.0`，`viscosity=1.0` |
| 状态方程 | 稳态不可压 NS | 稳态不可压 NS，调用 `StationaryIncompressibleNSLFEMModel` |
| 目标函数 | 论文内部流算例的主目标都是耗散最小化，即 `\int_\Omega \mu |\nabla u|^2 dA`，并且 `g=0` | 当前总目标是 `dissipation + volume_term + barycenter_term + regularization_term`，见 `objective.py` |
| 约束方式 | Pipe 和 T-Connection 都要求体积/面积保持；论文采用每次更新后的投影步骤 | 当前代码把体积和重心写成惩罚项；`volume_term`、`barycenter_term` 都由 `objective_parameters` 提供 |
| 优化算法 | 论文用的是固定步长的最速下降，`V = \alpha n`，没有 line search；更新后做体积/面积投影 | 当前代码默认是 `lbfgs`，带 Armijo line search，`initial_step_size=0.25`，`rtol=5e-4` |
| 形状导数 | 论文给出严格 Hadamard 标量密度 `G_NS` | 当前 `build_shape_derivative_source()` 仍是 nodewise traction-like 向量 + 面积/重心梯度，不是严格 Hadamard |

这张表说明一个很重要的事实：

1. **我们的物理问题方向和论文是一致的**，都是内部流、都在最小化黏性耗散；
2. **但是当前实现的数值策略和论文不完全一样**，尤其在出口条件、约束实现和优化算法上；
3. **如果要“直接按论文算例实现”，就不能只修 shape derivative，几何和优化流程也要一起对齐。**

再强调一下：

- 当前代码里的 `dissipation` 主项本身是和论文一致的，都是 `\mu \int_\Omega |\nabla u|^2 dA` 这一类耗散；
- 真正不一致的是 **shape derivative 的表达方式**、**出口边界处理**、**体积/面积约束的实现方式**，以及 **优化算法**。

---

## 13. 如果要按论文弯管算例直接实现，建议的参数模板

### 13.1 先复现论文 Pipe 算例

建议的目标设置是：

```math
J(u,\Omega) = \int_\Omega \mu \sum_{i,j=1}^d \left(\frac{\partial u_i}{\partial x_j}\right)^2 dA,
\qquad g=0.
```

建议的边界条件是：

```math
u=u^+ \text{ on } \Gamma_+,\qquad
u=0 \text{ on } \Gamma_0,\qquad
pn - \mu \partial_n u = 0 \text{ on } \Gamma_-.
```

建议的参数是：

| 参数 | 论文 Pipe 算例 |
| --- | --- |
| 黏度 `\mu` | `400` |
| 密度 `\rho` | `1.0` |
| 入口速度 | 抛物线分布，最大速度 `1.5` |
| 入口截面 | 1.0 |
| 雷诺数 | `Re=400` |
| 约束 | 体积保持 |
| 优化法 | 最速下降，固定步长 `V = \alpha n` |

### 13.2 再复现论文 T-Connection 算例

建议的目标仍然是：

```math
J(u,\Omega) = \int_\Omega \mu \sum_{i,j=1}^d \left(\frac{\partial u_i}{\partial x_j}\right)^2 dA.
```

建议的几何和约束是：

| 参数 | 论文 T-Connection 算例 |
| --- | --- |
| 入口位置 | 底部进入 |
| 出口位置 | 左右两个出口 |
| 固定边界 | 连接点后方一部分边界固定，保证必须转弯 |
| 入口速度 | 二次型（quadratic）入口 |
| 雷诺数 | `Re=100` |
| 约束 | 总面积保持 |
| 优化法 | 最速下降，固定步长，更新后投影 |

### 13.3 对应到我们当前代码时，最少要改的地方

如果要把当前弯管 benchmark 朝论文对齐，建议至少做下面这些改动：

| 当前代码项 | 论文对齐建议 |
| --- | --- |
| `objective.py` 的总目标 | 保留耗散主项，先把 `barycenter_term` 去掉或置零；体积约束最好改成投影而不是惩罚 |
| `build_shape_derivative_source()` | 把 traction-like 向量改成边界标量密度 `G_NS`，再乘 `n` 进入几何梯度 |
| `pressure_integral_target` | 如果要严格复现论文 pipe 算例，建议改回自然出流边界，而不是用压强积分归一化 |
| `algorithm="lbfgs"` | 如果要完全贴近论文，改成固定步长最速下降；如果先保留 `lbfgs`，那也要先保证 `G_NS` 是对的 |
| `volume_term` / `barycenter_term` | 论文里是“更新后投影”，不是“优化目标里加二次惩罚”；建议分开处理 |
| `state_system_is_linear` | 需要由 `fluid_model` 明确暴露，否则 NS 分支可能被错误当成线性系统 |

### 13.4 当前实现里的具体目标写法

当前代码实际对应的目标可以写成：

```math
J_{\mathrm{current}}
= J_{\mathrm{diss}}
+ J_{\mathrm{vol}}
+ J_{\mathrm{bar}}
+ J_{\mathrm{reg}}.
```

其中：

```math
J_{\mathrm{diss}} = \mu \int_\Omega |\nabla u|^2 dA,
```

```math
J_{\mathrm{vol}} = \frac{1}{2} \, \texttt{factor\_volume} \, (A - A_0)^2,
```

```math
J_{\mathrm{bar}} = \frac{1}{2} \, \texttt{factor\_barycenter} \, \|c-c_0\|^2,
```

```math
J_{\mathrm{reg}} = 0.
```

当前 `build_shape_derivative_source()` 里做的事情也可以概括成：

```math
\text{nodewise traction proxy}
\approx \mu (\nabla u + \nabla u^T)n - pn
```

再叠加：

```math
\text{area penalty gradient} + \text{barycenter penalty gradient}.
```

这就是为什么它“能跑”，但还不是论文意义上的严格 NS Hadamard 导数。
