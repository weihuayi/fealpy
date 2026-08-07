# fealpy | Issue｜Implementation｜SST k-omega 湍流模型有限元求解程序开发

## 一、Issue 基本信息

Issue ID：
- 8

Issue 类型（Issue Type）：
- `Implementation`

当前状态：
- `active`

状态候选值（当前为最小候选集，非最终冻结）：
- `draft`
- `active`
- `blocked`
- `pending-decision`
- `closed`

负责人：
- `李本桢`

指导人：
- `魏华祎`

协作者（按需填写）：
- `王鹏祥`

时间盒：
- `2026.3.17 - 2026.3.27`

开工分支：
- `implementation/issue-8-sst-k-omega-fem-solver`

相关 Model / Workflow / Template / Contract 引用（按需填写）：
- `fealpy/docs/design/hydraulic_pipe_fsi_optimization/sst_komega_solver_design.md`
- `tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_report.md` 

## 二、创建缘由与目标边界

创建缘由：
- 落实液压管件二次流与压损优化的先期调研，验证 $\text{SST} \,\, k-\omega$ 湍流模型在捕捉高压液压油二次流时的适用性。
- 根据设计文档，开发具备良好解耦性的有限元求解器模块，以支撑后续高压脉动载荷下的流固耦合（FSI）优化任务。

当前承载问题：
- 当前缺少标准化的、与底层网格及代数求解充分解耦的 $\text{SST} \,\, k-\omega$ 模型分离式有限元（FEM）求解程序。

目标：
- 完成 `Equation`（数学模型）、`Simulation`（离散算法）、`Math Model`（算例模型）和 `Computation Model`（计算模型）四个核心模块的代码编写与联调。
- 完成雷诺数 Re=43000 的 90° 弯管内部流动 Benchmark 算例并成功输出对比数据。

非目标：
- 本 Issue 不包含流固耦合（FSI）完整多目标优化的闭环控制（仅聚焦流体域基础求解器的单向开发）。
- 本 Issue 不包含对其它湍流模型（如 $k-\epsilon$ 或 LES）的开发支持。

## 三、范围、依赖与约束

工作范围：
- 实现 RANS 方程、 $k$ 方程、 $\omega$ 方程的经验常数管理与各项计算（生成项、耗散项等）。
- 实现这三个方程的有限元离散空间分配、刚度矩阵与右端项组装接口。
- 封装 `PipeBendTurbulentFlow` 算例边界解析闭包（入口抛物面速度、入口 k/ω 分布、出口静压）。
- 实现 `StationaryIncompressibleSSTkomegaFEMModel` 核心调度器的高频调度控制流与收敛残差检验逻辑。

前置依赖：
- FEALPy.CFD 基础框架组件可用。
- 高精度的 90° 弯管三维网格生成脚本或网格文件已就绪。

外部约束或限制：
- 程序架构必须严格遵循 FEALPy.CFD 的框架设计，确保物理边界、方程组装与底层网格空间的完全解耦。

当前不确定因素：
- 暂无

## 四、完成定义与风险

完成定义（DoD）：
- [ ] `equations/`、`simulation/`、`model/` 以及计算模型主文件代码全部提交并通过 Code Review 入库。
- [ ] 能够跑通 `PipeBendTurbulentFlow` 算例，且迭代残差满足收敛标准。
- [ ] 成功导出 VTK 流场数据，并能提取 75° 截面数据与参考文献的试验结果进行对比分析。

最小可接受结果：
- 算例能够正常启动分离式（Segregated）迭代，各组装和更新组件不报错，能输出合理趋势的流场初步结果。

主要风险：
- 分离式稳态迭代求解过程中容易发生数值发散（湍动能 $k$ 或比耗散率 $\omega$ 出现负值或奇异）。

回退或止损方式：
- 确保 benchmark 算例测试正确。

## 五、Gate System 概览

当前关键 Gate：
- `gate-sst-k-omega-solver-acceptance` （$\text{SST} \,\, k-\omega$ 求解器开发验收 Gate）

当前关键判断概览：
- **架构合规性裁决**：代码是否严格落实了模块化设计（`Equation` 仅负责经验常数与源项，`Simulation` 仅负责组装与分配，边界与网格完全隔离）？
- **数值收敛性裁决**：Re=43000 的 90° 弯管工况下，分离式迭代求解器能否稳定收敛，且 $k$ 和 $\omega$ 无严重负值振荡？
- **物理准确性裁决**：提取的 75° 截面流场二次流特征与截面数据，是否与基准文献数据趋势一致？
- **合入审查裁决**：PR 是否通过了协作者与指导人的 Code Review，且符合团队规范？

当前主要缺口：
- 代码实现待启动（PR 链接待生成）。
- Benchmark 验证证据待生成（残差收敛曲线图、75° 截面对比图、VTK 可视化结果待补充）。

知识抽取检查概览（按需填写）：
- 暂无

知识抽取相关关键入口（按需填写，仅列当前关键文件）：
- 暂无

知识抽取缺口或待补动作（按需填写）：
- 暂无

当前关键 Gate 文件入口（按需填写，仅列当前关键文件）：
- 暂无
## 六、后续动作与待确认项

建议下一动作：
- 执行启动与对齐动作，负责人确认底层接口依赖。

待确认项：
- 暂无

跨实例关系入口（按需填写）：
- 关联上位 Research Theme: `tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_theme_scope.md`
- 关联上位 Research report:`tiangong/kb/researches/hydraulic_pipe_fsi_optimization/tiangong_hydraulic_pipe_fsi_optimization_research_report.md` 
- 关联上位 Design:`fealpy/docs/design/hydraulic_pipe_fsi_optimization/sst_komega_solver_design.md`
