# fealpy | PR｜Design｜SST k-omega 模型有限元求解程序设计

## 1. 基本信息

- PR ID：<自动生成，如 #24>
- 上游 Issue ID：#3
- Issue 类型（Issue Type）：Design
- 当前状态：`pending-merge`
- 来源分支：`design/issue-3-sst-komega-solver-design`
- 目标分支：`develop`
- 远程 PR 链接：
- 相关治理资产引用：
  - `suanhai/models/suanhai_workflow_node_classification_model.md`
  - `suanhai/models/issue/suanhai_issue_model.md`
  - `suanhai/workflows/suanhai_issue_workflow.md`
  - `suanhai/workflows/suanhai_primary_value_workflow.md`
  - `suanhai/contracts/suanhai_commit_message_governance_contract.md`

## 2. 创建缘由与目标边界

### 2.1 创建缘由
承接《液压管路流固耦合与压损优化算法调研报告》的结论，为精确预测复杂管道内的分离流与壁面脉动压力，需引入 SST $k-\omega$ 湍流模型。本 PR 旨在将该模型的底层有限元求解器架构设计方案落盘，为正式编码阶段建立约束与蓝图。

### 2.2 当前 PR 解决什么
- 提交了 SST $k-\omega$ 模型求解器的架构设计文档（Equation / Simulation / Math Model / Computation Model）。
- 明确了支持 Re=43000 的 90° 弯管 Benchmark 算例验证的数据流设计。
- 提交了针对 `Design` 节点的 Gate 门禁裁决书记录。

### 2.3 当前 PR 不解决什么
- 不包含 SST $k-\omega$ 模型的具体有限元 Python 源码实现。
- 不包含流固耦合（FSI）固体域部分的设计。

### 2.4 与上游 Issue 的承接关系
本 PR 是对 Issue #3（Design 节点）工程价值输出的直接承载，满足 Issue #3 设定的全部完成定义（DoD）。

## 3. 变更摘要

### 3.1 主要变更
- `add`: 新增核心设计文档 `fealpy/docs/design/hydraulic_pipe_fsi_optimization/sst_komega_solver_design.md`。
- `add`: 新增门禁审计与授权记录 `fealpy/kd/issues/3/gate/decision.md`。

### 3.2 影响范围
- 文档域。不影响现有 FEALPy.CFD 的底层运行代码。

### 3.3 关键相关文件
- `fealpy/docs/design/hydraulic_pipe_fsi_optimization/sst_komega_solver_design.md`
- `fealpy/kd/issues/3/gate/decision.md`

### 3.4 非兼容变化
- 暂无。

### 3.5 主要风险
- 极低。当前仅为架构设计落盘，未对代码库产生影响。

### 3.6 回退方式
- 直接 Revert 本 PR 对应的合并提交即可。

## 4. Review 概览

### 4.1 当前 Review 状态
- Dod 已对齐，待评审。

### 4.2 主要评审意见摘要
- 架构解耦清晰，组件化策略符合 FEALPy 现有基础组件复用原则，基准算例支撑逻辑完备。

### 4.3 已解决项
- 程序设计符合当前 FEALPy.CFD 的框架。

### 4.4 未解决项
- 暂无。

### 4.5 当前阻断项
- 暂无。

## 5. 当前合并判断

### 5.1 当前合并判断

### 5.2 当前判断边界或适用范围
- 仅授权将 Issue #3 的“设计方案”资产合并至主干。

### 5.3 关键依据指针
- 裁决依据指向：`fealpy/kd/issues/3/gate/decision.md`。
  
### 5.4 对上游 Issue 推进的影响
- 本 PR 合并后，上游 Issue #3 的状态将正式流转为 `closed`。
- 授权开启新的 `Implementation` 类型 Issue 进入编码阶段。

### 5.5 需要同步回写的点
- 合并后需在 Issue #3 评论区或状态板中回写已合并的通知。

### 5.6 当前仍缺失的关键支撑
- 暂无。

## 6. 关键入口

### 6.1 关键 Commit
- `docs(design): add sst k-omega solver design and gate decision` 

### 6.2 关键讨论链接
暂无

### 6.3 关键验证结果入口
- [ ] 门禁裁决与审计证据库：`fealpy/kb/issues/3/gate/decision.md`（已通过验证逻辑走查，完成 DoD 审计）

### 6.4 关键相关文档入口
- 本次核心产出：`fealpy/docs/designs/sst_komega_solver_design.md`
- 上游输入凭证：`tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_report.md`

## 7. 对上游 Issue 的回写点

### 7.1 对上游 Issue 当前状态的影响
- 本 PR 合并后，上游 Issue #3 的状态需正式从 `pending-decision` 流转为 `closed`。

### 7.2 对 Gate System 的影响
- 确立 Issue #3 的 Gate 审计链条闭环。门禁文件作为唯一授权凭证正式落盘，结束当前主价值工作流节点（Design）的全部审批流程。

### 7.3 需要同步回写到主文件或索引的点
- 需要在 FEALPy.CFD 模块的设计文档索引页（如 `README.md` 或 `docs/index.md`）中，补充指向 `sst_komega_solver_design.md` 的链接。
- 在 Issue #3 的评论区回写本 PR 已合并的通知。

### 7.4 当前合并后仍未完成的事项
- 针对 Issue #3 本身，合并后即宣告 DoD 全部达成，无未完成事项。
- 针对更宏观的主价值推进流，当前仅完成“设计”落盘，尚未开始“编码”。

## 8. 后续动作与待确认项

### 8.1 建议下一动作
1. **执行合并**：由目标 Reviewer 确认后直接执行 Merge 操作。
2. **建立下游节点**：由负责人（李本桢）依据本设计文档，创建全新的 `Implementation` 类型 Issue，正式进入 `Simulation` 与 `Equation` 等模块的编码阶段。

### 8.2 待确认项
- 程序设计方案审计待确认。

### 8.3 触发下一轮判断更新的条件
- 当前已处于 Ready to Merge 状态。若在 Merge 动作执行前发现基础排版或合并冲突问题，则解决冲突后直接刷新本 PR 状态，否则不再触发判断更新。