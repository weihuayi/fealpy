# FEALPy | mesh_07 | 新网格模块验证缺陷台账

- **版本**：v0.1
- **状态**：待执行
- **入库位置**：`kb/developments/new_mesh_module/mesh_07_module_validation/mesh_07_module_validation_defect_ledger.md`
- **对应任务**：`mesh_07_module_validation`

本台账用于聚合成员验证记录中观察到的问题，维护唯一编号、重复关系、分类、严重度、复核状态和回流接口。台账项不自动表示 Bug 已确认，也不替代正式 Issue、修复记录或最终验收。

## 一、状态与分类口径

### 1.1 复核状态

| 状态 | 含义 |
|---|---|
| `reported` | 成员已报告，尚未独立复现 |
| `reproduced` | 非报告者已在记录基线或说明的等价环境中复现 |
| `confirmed` | 复现后已由相关维护者确认属于新网格模块或迁移缺陷 |
| `not reproduced` | 已尝试但尚未复现；保留尝试条件和差异 |
| `not a module bug` | 已复核为上层误用、环境、输入或其它非模块问题 |
| `duplicate` | 与另一唯一缺陷重复；必须回链主项 |
| `fixed pending revalidation` | 已有候选修复，等待原用例和回归测试复验 |
| `closed` | 修复、复验和必要回归保护完成，或经责任主体裁决关闭 |

### 1.2 类别

- `module bug`：新网格实现不符合明确接口、数学不变量或设计要求；
- `migration gap`：旧能力、算法或上层调用未完整迁移；
- `upper-layer adaptation`：上层 FEALPy 模块需要适配新网格接口；
- `documentation`：接口、迁移或使用说明存在缺失或歧义；
- `environment`：依赖、平台、后端或安装问题；
- `research-code issue`：成员研究代码或输入问题；
- `performance`：在可比较条件下出现阻断用途的明显性能退化；
- `unknown`：证据不足，尚不能分类。

### 1.3 严重度候选

| 严重度 | 建议口径 |
|---|---|
| `critical` | 静默产生广泛错误结果、数据损坏，或阻断绝大多数核心场景，建议暂停后续验收 |
| `high` | 阻断一个主要研究方向或核心上层模块，且无可靠规避方式 |
| `medium` | 影响特定操作或场景，但存在受限规避方式或影响范围较小 |
| `low` | 文档、诊断、易用性或非核心边界问题，不影响主要结果正确性 |
| `unknown` | 尚无足够证据分级 |

严重度只是候选值，最终口径由模块负责人及相关上层负责人复核。

## 二、本轮验证头信息

| 字段 | 内容 |
|---|---|
| `develop` 完整 commit hash | `<pending>` |
| 验证窗口 | `<pending>` |
| 团队成员总数 | `<pending>` |
| 适用成员数 | `<pending>` |
| 不适用成员数 | `<pending>` |
| 汇总人 | `<pending>` |
| 模块复核人 | `<pending>` |

## 三、缺陷台账

| 缺陷编号 | 问题摘要 | 来源报告 | 类别 | 严重度 | 复核状态 | 影响范围 | 建议回流 | 负责人 | 正式 Issue | 最近更新 |
|---|---|---|---|---|---|---|---|---|---|---|
| `M07-001` | `<summary>` | `<records/...#R-member-01>` | `<category>` | `<severity>` | `<status>` | `<scope>` | `<mesh_05 / mesh_06 / docs / upper layer / pending>` | `<owner or pending>` | `<URL or pending>` | `<YYYY-MM-DD>` |

> 执行开始前删除示例行；若本轮无问题，明确写“截至当前记录范围未观察到问题”，不得仅保留空表造成歧义。

## 四、单缺陷复核摘要

### `M07-001`：`<title>`

| 字段 | 内容 |
|---|---|
| 首次报告 | `<member record and report id>` |
| 重复报告 | `<other records or none>` |
| 受影响 commit / 环境 | `<hash and environments>` |
| 预期行为及依据 | `<expected + basis>` |
| 实际行为 | `<actual>` |
| 最小复现 | `<supporting/M07-001/...>` |
| 独立复现结果 | `<reviewer, attempts, result>` |
| 分类依据 | `<reason>` |
| 严重度依据 | `<reason>` |
| 临时规避与限制 | `<workaround or none>` |
| 建议承接对象 | `<implementation/test/docs/upper layer>` |
| 回归测试候选 | `<path or pending>` |
| 当前未决问题 | `<open questions>` |

## 五、重复与冲突记录

| 关系编号 | 来源问题 | 主缺陷或冲突对象 | 关系 | 处理说明 | 复核人 |
|---|---|---|---|---|---|
| `<DUP-001>` | `<report/defect>` | `<M07-###>` | `<duplicate / related / conflicting>` | `<why and how retained>` | `<reviewer>` |

## 六、回流与处置清单

| 缺陷编号 | 回流对象 | 所需动作 | 进入条件 | 完成反馈 | 当前状态 |
|---|---|---|---|---|---|
| `<M07-###>` | `<mesh_05/mesh_06/docs/upper layer/mesh_08>` | `<fix/test/document/decision>` | `<basis>` | `<required evidence>` | `<pending/in progress/done>` |

## 七、台账复核

| 检查项 | 结果 | 复核人 | 日期 | 备注 |
|---|---|---|---|---|
| 所有成员问题均已进入台账或有排除说明 | `<pending>` | `<reviewer>` | `<date>` | `<notes>` |
| 重复项保留所有来源回链 | `<pending>` | `<reviewer>` | `<date>` | `<notes>` |
| `confirmed` 项均有独立复现和维护者确认 | `<pending>` | `<reviewer>` | `<date>` | `<notes>` |
| 严重度与影响范围有依据 | `<pending>` | `<reviewer>` | `<date>` | `<notes>` |
| 重大风险均有负责人或待裁决状态 | `<pending>` | `<reviewer>` | `<date>` | `<notes>` |

## 附录：版本演进记录

- **v0.1**：
  - 变更人：AI Agent
  - 变更时间：2026-07-12
  - 变更摘要：首次建立缺陷分类、分级、复核和回流台账
