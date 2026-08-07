# Suanhai | Template | 任务收尾模板

- **版本**：v0.4
- **状态**：草案
- **入库位置**：`suanhaios/templates/task/suanhai_task_closure_template.md`
- **启用条件**：当某个具体 Task 进入正式收尾、归档与反向审计阶段时，本模板必须启用
- **适用范围**：SuanhaiOS 仓库；算海团队所有仓库中的 Task Closure 实例写作与修订场景

本模板对应文件 `suanhai_task_closure_template.md`，用于生成具体的 Task Closure 实例文档。该模板回答的问题是：当前 Task 如何正式收尾。

## 一、模板定位

本模板用于表达当前 Task 的正式收尾结果、经验沉淀与反向审计。本模板不替代 Validation，也不退化为一次性总结。

## 二、使用边界

- 回答如何正式收尾，不只做结果罗列。
- 明确稳定成果、未完成项、经验问题与反向审计结论。
- 不把执行日志、Validation 结果清单或过程复盘替代正文主位。
- 保持收尾对象清楚、结论可追溯、改进建议可执行。

## 三、模板正文骨架

```markdown
# <Repo> | Task Closure | <Task 中文标题> 收尾

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`<repo_relative_path>/<task_slug_paths>_task_closure.md`
- **启用条件**：当当前 Task 进入正式收尾与归档阶段时，本文件必须启用
- **适用范围**：<repo_or_scope>

本文档用于承载当前 Task 的正式收尾表达。

## 一、收尾对象与范围
本 Task 的收尾对象包括：

- `.vtu` 四面体网格读入能力
- tetra mesh 构造能力
- 半径比质量计算
- tetra 二面角计算
- 最小验证脚本
本 Task 属于 mesh_01_validation 中的 A→B→D 路径验证任务。

## 二、已形成的稳定成果
当前已经形成以下稳定成果：

### 1. vtu 文件读取能力

已基于 meshio 实现：

- `.vtu` 文件读取
- tetra cell 提取

并完成与新 mesh 架构对接。

---

### 2. tetra mesh 构造能力

已完成：

- MeshBlock 构造
- EntitySector 注册
- TopologyBuilder.construct 调用

能够正确建立 tetra 与 tri 拓扑关系。

---

### 3. 半径比质量计算

已完成：

- tetra 外接球半径计算
- tetra 内切球半径计算
- radius ratio 计算

能够稳定输出单元质量结果。

---

### 4. 二面角计算

已完成：

- tetra 面法向量计算
- 二面角计算
- 角度转换（degree）

能够输出 tetra 的 6 个二面角。

---

### 5. 最小验证流程

已形成：

```bash
python test_mesh.py --filename xxx.vtu

## 三、未完成项与原因

## 四、经验与问题

## 五、反向审计与后续建议

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
    - 首次建立当前 Task Closure
```

补充说明：模板标题中的 `<Repo>` 用于填写实例文档所属仓库在标题中的仓库标识。若实例文档属于 `suanhaios/`，则 `<Repo>` 写为 Suanhai；若实例文档属于 `whyos/`，则 `<Repo>` 写为 Why。

