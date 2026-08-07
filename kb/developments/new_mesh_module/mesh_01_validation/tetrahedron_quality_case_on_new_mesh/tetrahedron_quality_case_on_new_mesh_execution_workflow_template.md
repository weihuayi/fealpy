# Suanhai | Template | 任务执行工作流模板

- **版本**：v0.4
- **状态**：草案
- **入库位置**：`suanhaios/templates/task/suanhai_task_execution_workflow_template.md`
- **启用条件**：当某个具体 Task 已进入正式执行组织阶段时，本模板必须启用
- **适用范围**：SuanhaiOS 仓库；算海团队所有仓库中的 Task Execution Workflow 实例写作与修订场景

本模板对应文件 `suanhai_task_execution_workflow_template.md`，用于生成具体的 Task Execution Workflow 实例文档。该模板回答的问题是：这个 Task 的 How 如何被节点化、组织、推进、控制与收口。

## 一、模板定位

本模板用于定义当前 Task 的执行 How，是基于 Task 本体和 Task Network Design 提供的结构认知展开的执行 Workflow 表达。本模板不替代 Task 本体，不替代客观内在结构设计，不替代目标资产本体。

## 二、使用边界

- 回答 How，不回写 What。
- 明确 Workflow 的 Input、Node、Gate、Transition 与收口条件。
- 不把 Task 客观内在结构设计结果混写为执行步骤表。
- Node 设计应尽量达到可推进、可判定、可交付粒度。

## 三、模板正文骨架

```markdown
# <Repo> | Task Execution Workflow | <Task 中文标题> 执行工作流

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`<repo_relative_path>/<task_slug_paths>_task_execution_workflow.md`
- **启用条件**：当需要定义当前 Task 的执行 How 时，本文件必须启用
- **适用范围**：<repo_or_scope>

本文档用于定义当前 Task 的执行 How。

## 一、Workflow 定位
本 Workflow 用于组织四面体网格质量验证算例的实现流程，
验证新 mesh 架构在三维 tetra mesh 场景下的基础能力。

## 二、Workflow Input
输入包括：

- `.vtu` 四面体网格文件
- fealpy 新 mesh 模块
- meshio
- backend manager（bm）
## 三、Workflow Node 设计
### Node 1：读取 vtu 文件

目标：

- 使用 meshio 读取 `.vtu`
- 提取：
  - node
  - tetra cell

输出：

- 节点数组
- tetra 单元连接关系

---

### Node 2：构造 mesh

目标：

- 构造 MeshBlock
- 建立 tetra EntitySector
- 调用 TopologyBuilder.construct

输出：

- Mesh 对象
- tetra / tri 拓扑关系

---

### Node 3：计算几何量

目标：

- 计算：
  - face area
  - cell volume

输出：

- 面面积
- 单元体积

---

### Node 4：计算半径比

目标：

- 基于：
  - 外接球半径
  - 内切球半径

计算 tetra radius ratio。

输出：

- radius ratio array

---

### Node 5：计算二面角

目标：

- 计算 tetra 的 6 个二面角
- 使用法向量夹角计算

输出：

- dihedral angle array

---

### Node 6：结果输出与验证

目标：

- 输出：
  - radius ratio
  - dihedral angle
- 检查结果合法性

输出：

- 控制台结果
- 最小验证结论

## 四、Workflow Gate 与 Transition 设计
Gate 1：

- `.vtu` 成功读取后进入 mesh 构造阶段

Gate 2：

- mesh topology 构造成功后进入几何计算阶段

Gate 3：

- 几何量无异常后进入质量计算阶段

Gate 4：

- 所有指标正常输出后进入验证收口阶段


## 五、收口条件
满足以下条件即视为 Workflow 完成：

- `.vtu` 文件读取成功
- tetra mesh 正确建立
- 半径比成功计算
- 二面角成功计算
- 无 NaN / Inf
- 命令行脚本可正常运行


## 附录 A：本文件版本演进记录

- **v0.4**：
  - 变更人：魏华祎
  - 变更时间：2026-04-19
  - 变更摘要：
    - 在模板正文骨架后补充 `<Repo>` 仓库标识映射说明
    - 保持模板对象职责与正文骨架主位稳定

- **v0.1**：
  - 变更人：Wang Dong
  - 变更时间：<2026-05-06>
  - 变更摘要：
    - 首次建立当前 Task Execution Workflow
```