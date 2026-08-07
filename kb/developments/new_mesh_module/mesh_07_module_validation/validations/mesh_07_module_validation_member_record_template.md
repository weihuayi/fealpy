# FEALPy | mesh_07 | 成员研究工作验证记录模板

- **模板版本**：v0.1
- **对应任务**：`mesh_07_module_validation`
- **实例命名**：`records/<member_slug>_validation_record.md`
- **使用说明**：每位成员复制一份。存在基于 FEALPy 的研究工作时填写至少一个代表性用例；不存在时填写“不适用声明”，不得虚构用例。

> 本记录是 V&V 原始结果物，不是最终验收结论。所有判断仅覆盖本文件记录的 commit、环境、输入和用例。

## 一、成员与适用性

| 字段 | 填写内容 |
|---|---|
| 成员姓名 | `<name>` |
| member slug | `<member_slug>` |
| 提交日期 | `<YYYY-MM-DD>` |
| 是否有基于 FEALPy 的现有研究工作 | `<是 / 否>` |
| 研究方向或项目简称 | `<非敏感简介；不适用则填 N/A>` |
| 记录状态 | `<draft / submitted / reviewed>` |

### 不适用声明

仅在“是否有基于 FEALPy 的现有研究工作”为“否”时填写：

- 不适用原因：`<reason>`
- 本轮未运行研究用例，且本记录不构成对新网格模块的通过或失败判断。

## 二、验证基线与环境

| 字段 | 填写内容 |
|---|---|
| `develop` 完整 commit hash | `<40-char hash>` |
| 本地是否有附加修改 | `<否 / 是；若是列出文件和原因>` |
| 操作系统 | `<OS and version>` |
| Python 版本 | `<version>` |
| FEALPy 安装或运行方式 | `<editable/source/wheel + command>` |
| 后端与精度 | `<numpy/pytorch/jax/...; float32/float64/...>` |
| 关键依赖版本 | `<only relevant packages>` |
| 硬件信息（仅在相关时） | `<CPU/GPU/accelerator>` |

环境复现命令或依赖快照位置：

```text
<commands or path>
```

## 三、代表性研究用例

### Case `<member_slug>-01`：`<case title>`

| 字段 | 填写内容 |
|---|---|
| 用例目的 | `<该用例在研究工作中解决什么问题>` |
| 选择理由 | `<为何能代表日常 FEALPy 使用路径>` |
| 涉及 FEALPy 模块 | `<mesh/functionspace/fem/solver/...>` |
| 涉及网格类型和关键操作 | `<triangle/tetra/...; construct/boundary/measure/refine/...>` |
| 输入来源 | `<程序生成 / 脱敏数据 / 受控数据；说明生成方式>` |
| 预期行为或结果 | `<可判断的预期>` |
| 预期依据 | `<解析解、数学不变量、历史可信结果、收敛趋势或其它依据>` |
| 运行命令 | `<exact command>` |
| 运行状态 | `<PASS / FAIL / BLOCKED / INCONCLUSIVE>` |
| 开始与结束时间 | `<timestamps or duration>` |

关键配置：

```text
<configuration needed to reproduce>
```

实际结果摘要：

```text
<measured values, status, or concise output>
```

预期与实际比较：

| 观察量 | 预期 | 实际 | 容差或判断规则 | 状态 |
|---|---|---|---|---|
| `<observable>` | `<expected>` | `<actual>` | `<rule>` | `<match/deviation/unknown>` |

限制与未覆盖范围：

- `<limitation>`

> 如有多个代表性用例，复制本节并依次编号；无需为了数量增加无代表性的用例。

## 四、发现的问题

若无问题，保留表头并填写“本用例范围内未观察到问题”。

| 临时报告编号 | Case | 问题摘要 | 类别候选 | 严重度候选 | 状态 | 支撑材料 |
|---|---|---|---|---|---|---|
| `R-<member_slug>-01` | `<case-id>` | `<summary>` | `<module bug / migration gap / upper-layer adaptation / documentation / environment / unknown>` | `<critical/high/medium/low/unknown>` | `<reported/reproduced/confirmed/not reproduced/not a module bug>` | `<path/link>` |

### 问题 `R-<member_slug>-01`

- **首次失败命令**：`<exact command>`
- **预期行为**：`<expected behavior and basis>`
- **实际行为**：`<actual behavior>`
- **错误类型或异常**：`<exception/result deviation/performance/documentation>`
- **稳定复现情况**：`<x/y runs or not attempted>`
- **最小复现状态**：`<available / partial / unavailable>`
- **影响范围初判**：`<scope>`
- **临时规避方式**：`<none or workaround; do not hide side effects>`
- **敏感信息处理**：`<none / redacted / controlled review required>`

关键 traceback 或差异：

```text
<minimal relevant excerpt; do not paste secrets or unrelated full logs>
```

最小复现步骤或复现阻塞说明：

1. `<step>`
2. `<step>`
3. `<observed result>`

## 五、成员级结论

| 项目 | 填写内容 |
|---|---|
| 已执行用例数 | `<n>` |
| PASS / FAIL / BLOCKED / INCONCLUSIVE | `<counts>` |
| 报告问题数 | `<n>` |
| 当前研究用途判断 | `<supported / partially supported / not supported / inconclusive / not applicable>` |
| 判断适用边界 | `<case, environment, scale, backend>` |
| 是否需要立即升级 | `<no / yes + reason>` |

结论说明：

`<仅基于本记录事实进行说明，不将未覆盖范围写成通过。>`

## 六、复核记录

| 字段 | 填写内容 |
|---|---|
| 成员自查 | `<name, date, result>` |
| 完整性检查 | `<reviewer, date, result>` |
| 问题独立复现 | `<reviewer, date, result or N/A>` |
| 复核备注 | `<notes>` |

## 附录：版本演进记录

- **v0.1**：
  - 变更时间：2026-07-12
  - 变更摘要：首次建立成员研究工作验证记录模板
