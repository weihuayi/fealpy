## 一、Issue 基本信息

Issue ID：
- 3

Issue 类型（Issue Type）：
- `Design`

说明：
- 当前字段表示本 Issue 所承载的真实工作流节点名，不是装饰性分类标签

当前状态：
- `pending-decision`

状态候选值（当前为最小候选集，非最终冻结）：
- `draft`
- `active`
- `blocked`
- `pending-decision`
- `closed`

负责人：
- 李本桢

指导人：
- 魏华祎

协作者（按需填写）：
- 王鹏祥

时间盒：
- 2026.3.16 - 2026.3.17

开工分支：
- `design/issue-3-sst-komega-solver-design`

相关 Model / Workflow / Template / Contract 引用（按需填写）：
- suanhai/template/issue/suanhai_issue_main_template.md
- tiangong/kb/requirements/tiangong_hydraulic_pipe_fsi_optimization_scenario_requirement.md

## 二、创建缘由与目标边界

创建缘由：
- 根据《液压管路流固耦合与压损优化算法调研报告》的结论，为精确预测复杂管道内的分离流与壁面脉动压力， $\text{SST}\,\, k-\omega$ 模型被选定为目标流体求解算法。
- 在进入正式编码实现前，需要完成流体求解器的架构设计与模块划分，以确保后续开发路径清晰且可验证。

当前承载问题：
- 在有限元框架下，如何合理划分数学模型、离散算法、Benchmark 算例管理、矩阵方程解法器等功能模块？
- 如何设计程序接口与数据结构，以保证其能够稳定运行并正确求解调研报告中规划的 Benchmark 算例？

目标：
- 产出《 $\text{SST}\,\, k-\omega$ 模型有限元求解器程序设计文档》。
- 明确各个核心模块的职责边界与关键数据结构。
- 提供核心求解流程图。

非目标：
- 本 Issue 不包含任何实际的算法代码编写（编码工作将在后续的 Implementation Issue 中进行）。
- 不包含流固耦合（FSI）中固体域求解器的设计，仅聚焦于单一流体域的湍流程序架构。

## 三、范围、依赖与约束

工作范围：
- 稳态求解控制流程设计。
- 边界条件处理模块设计（特别是壁面函数的接口支持）。
- Benchmark 算例验证前置数据流设计（网格文件读取与后处理输出接口）。

前置依赖（按需填写）：
- Issue:https://github.com/suanhaitech/tiangong/issues/17
- 调研报告：tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_report.md。
- 主题范围：tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_theme_scope.md。

外部约束或限制（按需填写）：
- 模块设计必须符合团队现有计算架构的底层接口规范（如基础矩阵库、网格管理库的限制）。
- 架构设计必须具备可扩展性，以便未来挂载其他方程（如流固耦合界面数据传递）。

当前不确定因素（按需填写）：暂无。
  

## 四、完成定义与风险

完成定义（DoD）：
- 产出完整的程序设计文档，并提交 Pull Request。
- 设计方案中明确说明了如何支撑验证调研报告中规定的 Benchmark 算例。
- 核心功能模块具有清晰的类及接口定义，无明显的逻辑死锁或数据流断点。
- 通过团队的 Gate 评审（同行评审机制）。

最小可接受结果（按需填写）：
- 提供一份包含模块调用关系图与核心 API 说明的骨架设计文档，足以指导研发人员开展后续具体的编码工作。

主要风险（按需填写）：
- 湍流模型对流项的稳定化方法（如 SUPG/PSPG）如果在框架接口层面考虑不周全，可能导致后续实现时矩阵组装模块面临大规模重构风险。

回退或止损方式（按需填写）：
- 确保符合当前 FEALPy.CFD 的框架设计。

## 五、Gate System 概览

当前关键 Gate：
- `gate-program-design-review`

当前关键判断概览：
- 判断当前的程序设计方案是否合理、模块边界是否清晰、数据结构是否满足有限元求解需求。
- 判断该设计是否足以支撑后续正确且稳定地求解 $\text{SST}\,\, k-\omega$ 模型及 Benchmark 算例。

当前主要缺口：
- 尚未形成具体的程序设计草案。

知识抽取检查概览（按需填写）：
- 待检查：确保设计案中程序框架符合当前 FEALPy.CFD 的框架设计

知识抽取相关关键入口（按需填写，仅列当前关键文件）：暂无

知识抽取缺口或待补动作（按需填写）：暂无

当前关键 Gate 文件入口（按需填写，仅列当前关键文件）：
- `kb/issues/3/gate/decision.md`

## 六、后续动作与待确认项

建议下一动作：
- 确认本 Issue 的指派人员与排期时间盒。
- 负责人拉取分支 `design/issue-3-sst-komega-solver-design` 启动设计文档的起草。

待确认项：暂无

跨实例关系入口（按需填写）：暂无