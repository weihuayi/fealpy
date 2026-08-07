# fealpy | PR｜implementation｜三通管道几何与网格连通支持

## 1. 基本信息

- PR ID：待创建（创建后回填）
- 上游 Issue ID：15
- Issue 类型（Issue Type）：implementation
  - 说明：当前字段表示上游 Issue 所承载的真实工作流节点名，不是 PR 自己的装饰性分类标签
- 当前状态：pending-merge
  - 状态候选值：`draft` / `active` / `pending-merge` / `merged` / `closed`
- 负责人：ck_lab_wsl
- 评审人：待补充
- 源分支：`implementation/issue-15-tee-junction-pipe-geometry-mesh`
- 目标分支：`develop`
- 远程 PR 链接：待创建（创建后回填）
- 相关治理资产引用：
  - `suanhai/models/suanhai_workflow_node_classification_model.md`
  - `suanhai/models/issue/suanhai_issue_model.md`
  - `suanhai/workflows/suanhai_issue_workflow.md`
  - `suanhai/workflows/suanhai_primary_value_workflow.md`
  - `suanhai/contracts/suanhai_commit_message_governance_contract.md`

## 2. 创建缘由与目标边界

### 2.1 创建缘由

为支持三通管道场景的流固耦合（FSI）建模，需要在 `fealpy.mesher` 与 `fealpy.cgraph` 两层同时补齐可直接调用的三通网格能力，并提供可复现实例与最小测试入口。

### 2.2 当前 PR 解决什么

- 新增三通管道 FSI 网格器 `TeePipeMesher`，包含参数校验、几何构建、边界分类与局部网格场设置。
- 在 `fealpy.cgraph` 中新增 `TeePipeMesh` 节点类型，打通图计算方式下的三通网格生成。
- 增加命令行示例与 cgraph 示例，补齐用户使用路径。
- 补充用户手册与 cgraph 层单测，形成最小可回放材料。

### 2.3 当前 PR 不解决什么

- 不支持非等径三通（`D_main != D_branch`）。
- 不扩展现有 elbow 管道建模逻辑。
- 不在本 PR 内完成全量 gmsh 参数稳定性扫描与性能基准。

### 2.4 与上游 Issue 的承接关系

本 PR 是 Issue 15 的实现落地分支，承接上游关于“三通管道几何与网格”能力建设目标，提供可执行代码、示例与最小验证证据。

## 3. 变更摘要

### 3.1 主要变更

- 新增 `fealpy/mesher/tee_pipe_mesher.py`：实现三通管道 FSI 网格生成核心逻辑。
- 修改 `fealpy/mesher/__init__.py`：导出 `TeePipeMesher`。
- 修改 `fealpy/cgraph/mesh/hydraulic_pipe.py`：新增 `TeePipeMesh` 节点并导出。
- 新增 `example/mesher/tee_pipe_mesher_example.py`：命令行网格导出示例。
- 新增 `example/cgraph/tee_pipe_example.py`：cgraph 图式调用示例。
- 新增 `docs/design/hydraulic_pipe_fsi_optimization/fealpy_tee_pipe_mesher_user_manual_zh.md`：用户手册。
- 新增 `test/cgraph/test_hydraulic_pipe.py`：`TeePipeMesh` 参数透传与返回行为测试。

### 3.2 影响范围

- 受影响模块：`fealpy.mesher`、`fealpy.cgraph.mesh`、`example`、`docs/design`、`test/cgraph`。
- 相对 `develop` 的统计：8 files changed, 904 insertions(+), 1 deletion(-)。

### 3.3 关键相关文件

- `fealpy/mesher/tee_pipe_mesher.py`
- `fealpy/mesher/__init__.py`
- `fealpy/cgraph/mesh/hydraulic_pipe.py`
- `example/mesher/tee_pipe_mesher_example.py`
- `example/cgraph/tee_pipe_example.py`
- `test/cgraph/test_hydraulic_pipe.py`
- `docs/design/hydraulic_pipe_fsi_optimization/fealpy_tee_pipe_mesher_user_manual_zh.md`

### 3.4 非兼容变化

暂无明确非兼容变化；当前为增量能力引入。

### 3.5 主要风险

- gmsh OCC 圆角在极端参数下可能失败（`intersect_angle`、`R_fillet` 组合敏感）。
- 当前仅支持等径三通，若调用方误传非等径参数将触发 `NotImplementedError`。
- cgraph 示例依赖本地 gmsh 与 VTK 导出环境，环境缺失时可能运行失败。

### 3.6 回退方式

- 代码级回退：对本分支相对 `develop` 的增量提交执行回滚（优先按提交粒度回滚）。
- 文件级回退：移除新增三通相关文件，并恢复 `hydraulic_pipe.py` 与 `mesher/__init__.py` 到 `develop` 基线。

## 4. Review 概览

### 4.1 当前 Review 状态

待正式评审；当前为合并前整理状态。

### 4.2 主要评审意见摘要

暂无（待评审输入）。

### 4.3 已解决项

- 已完成三通 mesher 主实现、cgraph 节点接入、示例补齐、文档补齐、最小单测补齐。

### 4.4 未解决项

- 远程 PR 页面评审意见待补充。
- 更大范围回归测试结果待补充（当前仅有针对性测试证据）。

### 4.5 当前阻断项

暂无硬阻断；主要依赖评审通过与远程 PR 流程完成。

## 5. 当前合并判断

### 5.1 当前合并判断

建议进入 `develop` 合并流程（`pending-merge`）。

### 5.2 当前判断边界或适用范围

该判断基于本次增量改动范围与最小验证结果，不代表上游 Issue 15 的全部收口完成。

### 5.3 关键依据指针

- 分支差异（`develop...implementation/issue-15-tee-junction-pipe-geometry-mesh`）显示新增三通能力链路完整。
- 本地验证：`python -m pytest test/cgraph/test_hydraulic_pipe.py -q` 通过（1 passed）。

### 5.4 对上游 Issue 推进的影响

该 PR 将 Issue 15 从“方案/实现中”推进到“可评审可合并”的代码形态。

### 5.5 需要同步回写的点

- PR 创建后需把 PR ID、远程链接、评审结论、最终合并结论回写到 Issue 15 主文件/索引。

### 5.6 当前仍缺失的关键支撑项

- 远程 PR 的评审记录与结论。
- 合并后在 `develop` 的复测记录（如需）。

## 6. 关键入口

### 6.1 关键 Commit

- `584ac7e16` `feat(mesher): add tee pipe mesher and example with doc.`
- `f8c8b9839` `feat(cgraph): add graph node of tee pipe`
- `3b3101acb` `docs: add issue file for #15`

### 6.2 关键讨论链接

暂无（待补充）。

### 6.3 关键验证结果入口

- 本地命令：`python -m pytest test/cgraph/test_hydraulic_pipe.py -q`
- 结果：`1 passed in 0.80s`（2026-04-02）

### 6.4 关键相关文档入口

- `docs/design/hydraulic_pipe_fsi_optimization/fealpy_tee_pipe_mesher_user_manual_zh.md`
- `kb/issues/15/fealpy_issue_15.md`

## 7. 对上游 Issue 的回写点

### 7.1 对上游 Issue 当前状态的影响

Issue 15 的实现材料已形成可评审 PR 载体，可进入评审与合并阶段。

### 7.2 对 Gate System 的影响

本 PR 提供实现与最小验证证据输入，但不等同于 Gate 全部通过。

### 7.3 需要同步回写到主文件或索引的点

- PR ID、远程链接、评审通过结论、合并时间与目标分支。

### 7.4 当前合并后仍未完成的事项

- 非等径三通支持（后续议题）。
- 参数稳定性与更大范围自动化验证补充（后续议题）。

## 8. 后续动作与待确认项

### 8.1 建议下一动作

1. 创建远程 PR（源分支 -> `develop`）并回填 PR ID/链接。
2. 发起评审并收敛意见，必要时补充回归测试证据。
3. 合并后回写 Issue 15 主文件与索引状态。

### 8.2 待确认项

- 评审人名单。
- 远程 PR 链接与 PR 编号。
- 是否需要在合并前增加 gmsh 参数稳定性测试样例。

### 8.3 触发下一轮判断更新的条件

- 评审意见新增或阻断项出现。
- 新验证结果与当前结论不一致。
- PR 状态发生变化（merged/closed）。
