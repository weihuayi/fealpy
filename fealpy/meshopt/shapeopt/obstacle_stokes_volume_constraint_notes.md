# Obstacle Stokes Volume Constraint Notes

> 这份笔记整理 obstacle Stokes benchmark 的完整数学链路，并把 cashocs 论文/实现与我们当前 FEALPy 代码逐层对齐。重点结论是：无体积约束时链路是稳定的；体积约束的导数公式经过最小算例验证是自洽的，但强罚项下的有限步更新会让目标函数变硬，导致“看起来像方向反了”的现象。

## 1. 当前结论

我们现在对 obstacle Stokes 的判断已经比较清楚：

- **无体积约束时**，优化链路可以稳定走通，目标值单调下降，几何形变也符合预期。
- **加入体积约束后**，如果 `factor_volume` 太小，约束效果不明显；如果 `factor_volume` 太大，第一步或后续几步就可能出现目标值上升，甚至整条轨迹被罚项主导。

这并不等价于“体积项导数符号错了”。我们已经做过最小几何验证，结论是：

- `signed_area` 的符号只取决于多边形节点的绕行顺序；
- 体积项的解析梯度和有限差分一致；
- 圆孔多边形顺时针 / 逆时针顺序下，梯度方向都能自洽。

因此，当前问题更像是：

- **体积罚项的强度与当前离散形变链不匹配**
- **有限步更新、线弹性传播、PDE 重解共同作用，使 objective landscape 变得很“硬”**

## 2. 无体积约束时为什么能走通

无体积约束时，目标主要由耗散项驱动：

\[
J(\Omega, u) = \mu \int_{\Omega^{flow}} \|\nabla u\|_F^2 \, dx
\]

这条链之所以能稳定工作，主要因为：

1. **PDE 主目标方向本身是稳定的**
   - obstacle Stokes 的流场、伴随场、边界 traction 可以形成一致的 shape derivative 主链。

2. **shape derivative 最终统一汇入同一个出口**
   - `objective.py` 是总 shape derivative 的统一入口。
   - `geometry_gradient.py` 只负责投影和传播，不再额外拼散乱的正则梯度。

3. **线弹性传播在没有额外强罚项时足够平滑**
   - 传播后内部网格虽然会变形，但通常不会把方向彻底扭坏。

4. **没有额外“硬峡谷”**
   - 因而固定步长或者带 Armijo 的 line search 更容易接受试探步。

换句话说，无体积约束时，主目标梯度和几何传播链是相对匹配的，因此结果稳定。

## 3. 体积约束的完整数学形式

当前 obstacle benchmark 的总目标可写成：

\[
J(\Omega, u)
=
\mu \int_{\Omega^{flow}} \|\nabla u\|_F^2 \, dx
+
\frac{v_1}{2}\,(V(\Omega_{\text{obs}})-V_0)^2
+
\frac{v_2}{2}\,\|C(\Omega_{\text{obs}})-C_0\|_2^2
\]

其中：

- \(\mu\) 是流体黏度，对应 `viscosity`
- \(v_1\) 是体积罚项权重，对应 `factor_volume`
- \(v_2\) 是重心罚项权重，对应 `factor_barycenter`
- \(V_0\) 是初始障碍体积
- \(C_0\) 是初始障碍重心

在当前代码里，这三部分分别对应：

- [objective.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/objective.py)
- [geometry_regularization.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py)
- [benchmark_obstacle_stokes.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/benchmark_obstacle_stokes.py)

## 4. 体积项在当前实现里是怎么构造导数的

体积项的导数不再是旧的 finite-difference fallback，而是统一由 `ObstacleGeometryRegularization` 构造。

### 4.1 几何量的计算

体积和重心通过多边形鞋带公式计算：

\[
A = \frac{1}{2}\sum_i (x_i y_{i+1} - x_{i+1} y_i)
\]

- 当前体积：

\[
V = |A|
\]

- 当前重心：

\[
C = \frac{1}{6A}\left(\sum_i (x_i + x_{i+1})\,\text{cross}_i,\;\sum_i (y_i + y_{i+1})\,\text{cross}_i\right)
\]

对应代码位置：

- [geometry_regularization.py#L77](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py#L77)
- [geometry_regularization.py#L184](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py#L184)

### 4.2 体积导数

体积罚项为：

\[
J_{\text{vol}} = \frac{v_1}{2}(V - V_0)^2
\]

其导数在当前实现中写成：

\[
\nabla J_{\text{vol}} = v_1 (V - V_0)\,\mathrm{sign}(A)\,\nabla A
\]

对应代码：

- [geometry_regularization.py#L279-L284](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py#L279)

这里的 `sign(A)` 是为了把有符号面积导数转换成绝对体积导数的物理方向修正。

### 4.3 重心导数

重心罚项为：

\[
J_{\text{bar}} = \frac{v_2}{2}\|C - C_0\|_2^2
\]

其导数由面积与一阶矩的商法则给出：

\[
\nabla C_x = \frac{\nabla M_x \cdot A - M_x \cdot \nabla A}{A^2},
\qquad
\nabla C_y = \frac{\nabla M_y \cdot A - M_y \cdot \nabla A}{A^2}
\]

对应代码：

- [geometry_regularization.py#L289-L293](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py#L289)

## 5. 为什么 `factor_volume` 大了会“反而上升”

这是当前最需要谨慎解释的地方。

我们已经验证过：

- `signed_area` 没有算反；
- 体积项梯度与有限差分一致；
- 圆孔测试也通过；
- 在极小步长下，体积项对应的局部方向是下降的。

这说明“体积导数符号错了”**不是**当前问题的主因。

但是当 `factor_volume` 很大时，会出现下面的数值效应：

1. **罚项变得非常硬**
   - \( \frac{v_1}{2}(V - V_0)^2 \) 的曲率随 `v1` 线性增大。
   - 一旦几何偏离参考体积，目标函数在体积方向上会变得很陡。

2. **有限步更新不再只看局部下降方向**
   - 虽然局部梯度方向仍然正确，但一次有限步更新会经过：
     - 边界位移
     - 线弹性传播
     - 试探网格生成
     - 状态方程重解
     - 新 objective 计算
   - 这些步骤都会放大非线性效应。

3. **PDE 主目标与罚项之间会形成“硬拉扯”**
   - 纯 PDE 方向往往仍然是下降的。
   - 但一旦强体积罚项加入，合成方向会被旋转，实际试探步在有限步长下很容易跨入上升区。

因此，当前现象更像是：

- **导数局部正确**
- **但强罚项 + 有限步更新 + 几何传播链，使得实际可接受步很窄**

这也是为什么：

- `v1` 太小，看不出约束效果；
- `v1` 太大，objective 很容易在前几步就变坏。

## 6. 已验证的梯度检查

我们做过两个最关键的验证。

### 6.1 纯几何最小算例

脚本：

- [debug_volume_regularization.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/debug_volume_regularization.py)

它验证了：

- 普通三角形 `ccw` / `cw`
- 圆孔近似多边形 `ccw` / `cw`

结果都显示：

- `signed_area` 会随节点顺序变号；
- `current_volume` 始终取绝对面积；
- 解析梯度与有限差分一致，误差在 `1e-9 ~ 1e-11` 量级；
- 因此当前体积项的几何符号链是自洽的。

### 6.2 obstacle benchmark 上的极小步验证

在 obstacle Stokes 的真实 benchmark 上，我们还做过极小步验证：

- `factor_volume = 1e4`
- `factor_barycenter = 1e2`
- `step_size = 1e-6`

结果显示：

- 目标值是下降的；
- 说明体积项的局部方向不是反向；
- 但在较大固定步长下，目标就可能上升。

这进一步支持了当前判断：

- **体积项的导数公式本身没错**
- **问题主要在“罚项太硬 + 有限步更新 + 非线性传播”**

## 7. 为什么无体积约束能走通，而体积约束会出问题

这两个现象可以同时成立，并不矛盾。

### 无体积约束

- 主目标较平滑；
- shape derivative 主链比较自然；
- 传播后不会形成很窄的峡谷；
- 因而优化稳定。

### 有体积约束

- 目标函数多了一项强二次罚；
- 该项在局部方向上是对的，但曲率可能很大；
- 有限步更新后，体积项会迅速主导局部 objective 变化；
- 因而看起来像“方向翻了”，实际上更像是“可接受区域变窄了”。

## 8. 当前还没有完全解决的问题

虽然体积项的几何导数检查通过了，但 obstacle Stokes 带体积约束时仍然存在一个现实问题：

- `factor_volume` 很大时，优化轨迹容易上升；
- `factor_volume` 很小时，体积约束效果不明显。

这意味着：

1. 体积导数公式**不是主要 bug**
2. 但当前数值链对强罚项的容忍度还不够
3. 可能仍需要进一步处理：
   - 权重归一化
   - 更稳健的步长 / 线搜索
   - 或者更接近 cashocs 的投影与更新闭环

## 9. 当前能够稳定跑通的 Stokes 完整链路

这一节专门总结 **不加体积约束时**，obstacle Stokes 为什么能稳定跑通。它不是某一个局部环节“碰巧对了”，而是状态、伴随、梯度、步长和重网格这几层已经形成了一个局部自洽的链路。

### 9.1 几何、边界和自由度是先对齐的

当前 obstacle benchmark 的几何可以理解为：

- 外边界固定；
- 障碍是可变的 hole；
- inlet 给定速度；
- obstacle 和 wall 无滑移；
- outlet 自然出流。

这些设置都在 [benchmark_obstacle_stokes.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/benchmark_obstacle_stokes.py) 里组织。

其中几个关键点是：

- 边界节点按角度排序，障碍边界有稳定绕行顺序；
- 设计边界和固定边界不会混掉；
- remesh 后会重新挂回边界角色和设计边界顺序。

这一步决定了后续体积、重心和 shape derivative 到底是对哪个几何对象起作用。

### 9.2 状态求解先把 Stokes 闭合住

状态方程是整个链路的基础：

\[
-\Delta u + \nabla p = 0,\qquad \nabla \cdot u = 0
\]

配上：

- inlet 速度边界；
- obstacle / wall 无滑移；
- outlet 自然边界；
- 压力零空间处理。

当前能稳定跑通的版本里，压力处理已经收敛到更稳的方式：**不再用 mean-zero 增广去硬钉整场压力，而是 pin 一个 pressure dof 来消去零空间**。这和 cashocs 的实际数值行为更接近，也避免了之前 mean-zero 增广下速度场被带偏的问题。

对应模块是：

- [state_solver.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/state_solver.py)
- [benchmark_obstacle_stokes.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/benchmark_obstacle_stokes.py)

### 9.3 目标函数先统一刷新，再进入导数链

当前目标值的统一入口是 [objective.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/objective.py)。

无体积约束时，实际目标几乎就是耗散项：

\[
J(\Omega, u) = \mu \int_{\Omega^{flow}} \|\nabla u\|_F^2 \, dx
\]

当体积 / 重心项打开时，它们也会先刷新当前几何量，再进入同一个目标值和 shape derivative 入口。也就是说，目标值不是“状态方程一套，正则一套，各算各的”，而是统一在一个 `objective_result` 里组织。

### 9.4 伴随求解把主目标的敏感性抽出来

伴随方程的作用，是把主目标对状态的敏感性转成边界上的 shape derivative 来源。

当前链路里，伴随求解主要由 [adjoint_solver.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/adjoint_solver.py) 负责。它会：

- 从当前 state 和 objective 构造伴随右端；
- 对右端做必要的归一化或一致化处理；
- 在需要时给出回退路径，但回退路径现在也尽量和共享几何正则保持同一口径。

对于无体积约束的稳定版本来说，伴随方程的角色很直接：

- 它只负责把 PDE 主目标的梯度信息抽出来；
- 不额外引入强罚项；
- 因而它给出的方向通常是平滑且可用的。

### 9.5 梯度构造是统一出口，而不是多处拼补丁

梯度构造现在有两个来源：

1. PDE 主目标的边界 traction 贡献；
2. 几何正则的贡献（在无体积约束时这里就是 0）。

它们最终都通过 `objective.py` 合并，形成一个统一的 `shape_derivative_source`。对应模块分工是：

- [benchmark_obstacle_stokes.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/benchmark_obstacle_stokes.py) 提供 PDE 牵引项；
- [geometry_regularization.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_regularization.py) 提供几何正则项；
- [objective.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/objective.py) 做统一合并；
- [geometry_gradient.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/geometry_gradient.py) 只负责把统一导数转成边界位移 / 控制向量。

这也是无体积约束时能够稳定跑通的一个核心原因：**方向没有被多层后处理重复改写**。

### 9.6 下降优化与步长检查只看最终试探态

优化器这一层在 [shape_optimizer.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/shape_optimizer.py) 里。

它的职责不是重新理解 PDE，而是只做一件事：

1. 给出当前下降方向；
2. 生成试探步；
3. 计算试探态的 objective；
4. 按 Armijo / 接受准则判断是否收下这一步。

当前稳定跑通的无体积版本，之所以不会乱掉，主要因为：

- PDE 主目标方向本身局部下降；
- 没有强体积罚项去把曲率拉得特别硬；
- 所以 line search 看到的试探值通常是可接受的。

如果改成固定步长，也能跑通，是因为这时目标地形相对平缓，步长不会把方向立刻推入上升区。

### 9.7 重网格是保护机制，不是主链路的主导因素

最后一步是网格传播和质量检查，对应 [mesh_propagation.py](/D:/study/github_repositories/fealpy/fealpy/meshopt/shapeopt/cashocs_implement/mesh_propagation.py)。

它的工作顺序大致是：

1. 把边界位移通过线弹性 / 扩展算子传到内部；
2. 检查单元质量；
3. 如果出现负单元或质量过差，就触发 remesh；
4. remesh 后重新挂回边界角色和设计边界顺序。

无体积约束时，这一层通常只是“把边界动作安全地扩展到内部”：

- 位移幅度不至于太硬；
- 单元质量大多保持在 good；
- remesh 很少成为主导因素。

因此，**当前能跑通的真正原因不是 remesh 很强，而是前面的状态、伴随、梯度和步长已经形成了一条局部自洽链，mesh propagation 只是把这条链安全地落到实际网格上。**

## 10. 与 cashocs 的关系

cashocs 在这类体积 / 重心正则上更稳定，主要因为它把正则项放在同一变分框架中处理：

- 每步刷新几何量；
- 用解析 shape derivative；
- 再投影到同一个控制空间；
- 由 L-BFGS / Armijo 做试探与接受。

我们现在已经做到的接近点是：

- 共享几何正则对象；
- 每步刷新当前几何量；
- 体积 / 重心导数的解析链式法则；
- shape derivative 统一入口。

但仍然保留 FEALPy 自己的离散传播链，所以在强罚项下更容易表现出敏感性。

## 11. 小结

当前最可靠的结论是：

- **无体积约束的 obstacle Stokes benchmark 是正确的，并且已经能稳定走通。**
- **体积约束的导数公式和符号链经过最小测试是正确的。**
- **但大 `factor_volume` 下的优化行为仍然不稳定，这更像是数值链路与强罚项不匹配，而不是体积导数公式本身错误。**

后续如果要继续排查，建议优先看：

1. 体积罚项是否需要无量纲化；
2. 是否需要更贴近 cashocs 的投影 / 更新闭环；
3. 是否需要对 `factor_volume` 做 continuation，而不是一开始就取很大。
