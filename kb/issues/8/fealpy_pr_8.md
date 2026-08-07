# fealpy | PR｜Implementation｜阶段性提交传统 NS 方程 90° 弯管基准算例求解器

## 1. 基本信息

- PR ID：19
- 上游 Issue ID：8
- Issue 类型（Issue Type）：`Implementation`
  - 说明：当前字段表示上游 Issue 所承载的真实工作流节点名，不是 PR 自己的装饰性分类标签
- 当前状态：`active`
  - 状态候选值：`draft` / `active` / `pending-merge` / `merged` / `closed`
- 负责人：李本桢
- 评审人：魏华祎、王鹏祥
- 源分支：`implementation/issue-8-sst-k-omega-fem-solver`
- 目标分支：`develop`（或对应基线分支，请确认）
- 远程 PR 链接：https://github.com/suanhaitech/fealpy/pull/19
- 相关治理资产引用：
  - `suanhai/models/suanhai_workflow_node_classification_model.md`
  - `suanhai/models/issue/suanhai_issue_model.md`
  - `suanhai/workflows/suanhai_issue_workflow.md`
  - `suanhai/workflows/suanhai_primary_value_workflow.md`
  - `suanhai/contracts/suanhai_commit_message_governance_contract.md`

## 2. 创建缘由与目标边界

### 2.1 创建缘由
原上游 Issue 8 旨在开发 $\text{SST} \,\, k-\omega$ 模型分离式有限元求解程序。但在推进过程中发现，当前 $\text{SST} \,\, k-\omega$ 模型算法调研不充分，计算结果不稳定且发散风险高，触发了原定的风险应对与止损机制（即先确保 benchmark 算例能跑出结果）。
为避免阻塞后续高压脉动载荷下的流固耦合（FSI）验证阶段的开展，本次 PR 优先提交基于传统 N-S（Navier-Stokes）方程、理论更为扎实的 90° 弯管内流场仿真结果与求解代码，作为本次任务的阶段性成果。

### 2.2 当前 PR 解决什么
当前 PR 提供了一个能够正常启动且无组装报错的 90° 弯管流场求解器（基于传统 N-S 方程），并成功导出了合理的流场仿真数据，直接支撑下一阶段 FSI 流固耦合验证的开展。

### 2.3 当前 PR 不解决什么
当前 PR 不包含 $\text{SST} \,\, k-\omega$ 模型下 $k$ 方程与 $\omega$ 方程的稳定离散求解器。针对算法数值发散（湍动能 $k$ 或比耗散率 $\omega$ 出现负值或奇异）的问题，不在本 PR 内解决。

### 2.4 与上游 Issue 的承接关系
本 PR 属于上游 Issue 8 `Implementation` 节点推进过程中的“止损降级阶段产物”。按照工作流契约，本阶段性代码合入后，由于原始 DoD 中 $\text{SST} \,\, k-\omega$ 相关的核心需求未能完全满足，Issue 8 将进入 **拆分（Split）** 流程，把未完成的 $\text{SST} \,\, k-\omega$ 算法开发任务抽取并转入全新的 Issue 进行后续攻坚。

## 3. 变更摘要

### 3.1 主要变更
- 实现了基于传统 N-S 方程的三维有限元离散空间分配、刚度矩阵与右端项组装接口。
- 封装了基于 N-S 方程的 `PipeBendTurbulentFlow` 算例边界解析闭包（入口速度、出口静压等）。
- 支持 VTK 流场数据的成功导出与输出。

### 3.2 影响范围
本次提交主要在求解器模块新增了传统 N-S 方程的稳态算例与对应接口，与原有的 FEALPy.CFD 基础框架及网格生成部分保持解耦，未引入非兼容性的核心框架破坏。

### 3.3 关键相关文件
- `equations/`（涉及 N-S 方程的基础经验常数管理与各项计算逻辑实现）
- `simulation/`（涉及传统分离式算法组装和更新组件）
- 计算模型主文件及 90° 弯管 Benchmark 测试算例文件

### 3.4 非兼容变化
暂无。

### 3.5 主要风险
传统的 N-S 方程算法无法精确捕获 Re=43000 高压液压油工况下复杂的二次流现象特征。该计算结果仅用作 FSI 耦合流程跑通的首期验证替代品，不能直接用于最终流场物理特性的文献对标分析。

### 3.6 回退方式
撤销当前合并的 N-S 方程求解模块提交记录，恢复计算模型与框架至合并前状态。

## 4. Review 概览

### 4.1 当前 Review 状态
等待评审。

### 4.2 主要评审意见摘要
（暂无，待 Review 后补充）

### 4.3 已解决项
（暂无，待 Review 后补充）

### 4.4 未解决项
（暂无，待 Review 后补充）

### 4.5 当前阻断项
无。当前代码已满足最小止损与替代支持标准。

## 5. 当前合并判断

### 5.1 当前合并判断
建议合并。以确保流固耦合相关的测试工作不被算法开发进度阻塞。

### 5.2 当前判断边界或适用范围
仅保证传统 N-S 方程对弯管结构的代数计算不发生数值发散，能输出具备合理宏观趋势的初步流场结果，为后续代码提供接口承接依据。

### 5.3 关键依据指针
- 弯管仿真算例残差曲线
- VTK 流场结果可视化截图（待补充至当前 PR 评论区）

### 5.4 对上游 Issue 推进的影响
本 PR 的合并将直接触发上游 Issue 8 的拆分（Split）动作。合并完成后，当前 Issue 8 将完成最小回写、归档（Archive）并随后关闭（Close）。

### 5.5 需要同步回写的点
必须回写 Issue 8 的主文件，修改其实际的完成定义（DoD）兑现情况。需在 `gate/decision.md` 裁决记录中说明“降级采用 N-S 方程”的原因。

### 5.6 当前仍缺失的关键支撑项
高雷诺数复杂流场捕获能力的理论论证及 $\text{SST} \,\, k-\omega$ 模型程序的稳定实现方案。

## 6. 关键入口

### 6.1 关键 Commit
- https://github.com/suanhaitech/fealpy/commit/d17bd3aa795e98637b22c0a2bc1e644c8f31734e

### 6.2 关键讨论链接
暂无

### 6.3 关键验证结果入口
- 弯管稳态流场迭代残差曲线及截面速度场 VTK 结果（见附件或当前 PR 讨论区）。

### 6.4 关键相关文档入口
- `fealpy/docs/design/hydraulic_pipe_fsi_optimization/sst_komega_solver_design.md`

## 7. 对上游 Issue 的回写点

### 7.1 对上游 Issue 当前状态的影响
Issue 将从 `active` 转入 `split`（拆分），最终流向 `archive` 与 `close`。

### 7.2 对 Gate System 的影响
`gate-sst-k-omega-solver-acceptance` 的原定通过条件失效，需根据实际调整的策略判定为“条件性通过（偏离原定模型）”。需要在裁决区明文记录由于发散问题退回基础 N-S 方程的决策逻辑。

### 7.3 需要同步回写到主文件或索引的点
在 Issue 8 的完成情况中清晰标注：条目1（架构合规）、条目2（降级跑通算例）达标；条目3（二次流提取与基准对标分析）延期。

### 7.4 当前合并后仍未完成的事项
$\text{SST} \,\, k-\omega$ 的 $k$ 方程和 $\omega$ 方程在分离式稳态求解中的数值发散阻击策略。

## 8. 后续动作与待确认项

### 8.1 建议下一动作
- 指导人与协作者完成当前 PR 的代码合规性与边界隔离 Review。
- 执行 Issue 拆分，由负责人员建立全新的 Issue 专门追踪 $\text{SST} \,\, k-\omega$ 算法调研。

### 8.2 待确认项
- 拆分出来的新 Issue 的预期时间盒与责任分配。
- 下一阶段基于当前 N-S 输出对接 FSI 工具链的具体验证时间点。

### 8.3 触发下一轮判断更新的条件
评审人提交 Code Review 意见。