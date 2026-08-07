# mesh_schema_interface_test_refactor

将新网格 Schema 测试从按形状组织重构为按标准接口和算法组织的独立任务。

## 任务目标

建立统一的跨形状测试数据集合，以及唯一的 Schema 标准接口测试文件。测试文件按 Schema 标准接口分组，通过参数化覆盖不同形状、维数、规则/非规则几何、多实体和索引子集场景。

本任务不负责修改生产 Schema 实现；测试发现的生产缺陷通过可复现 case 回流到对应实现任务。

## 已启用资产

| 资产 | 用途 |
|---|---|
| `mesh_schema_interface_test_refactor_task_requirement.md` | 定义需求、约束、成功条件和待核验事项 |
| `mesh_schema_interface_test_refactor_task_brief.md` | 稳定任务定位、输入输出和完成判据 |
| `mesh_schema_interface_test_refactor_task_object_boundary.md` | 区分测试重构、生产修复、拓扑/mesher/IO 测试和验收 |
| `mesh_schema_interface_test_refactor_task_target_asset_design.md` | 设计测试数据、唯一测试入口、覆盖矩阵和迁移记录 |
| `mesh_schema_interface_test_refactor_task_verification_and_validation_design.md` | 区分测试实现正确性与重构后可维护性确认 |
| `mesh_schema_interface_test_refactor_task_execution_workflow.md` | 组织盘点、迁移、对照、旧测试处置和回归收口 |

## 未启用模板

- **Task Network Design**：本任务作为新网格模块测试工作的独立承接项，当前先建立任务自身资产；是否纳入总任务网络另行裁决，不在本次擅自修改父网络。
- **Task Agent Prompt**：当前任务面向开发和测试人员执行，不要求统一 AI Agent 自主执行提示词。
- **Task Closure**：任务尚未执行，没有实际记录、复核结果或收口证据，暂不建立 Closure。

## 预期核心实现资产

- `tests/mesh/data/schema_cases/`
- `tests/mesh/data/schema_cases/__init__.py`
- `tests/mesh/data/schema_cases/common.py`
- `tests/mesh/data/schema_cases/simplex.py`
- `tests/mesh/data/schema_cases/tensor_product.py`
- `tests/mesh/data/schema_cases/mixed.py`
- `tests/mesh/data/schema_cases/references.py`
- `tests/mesh/unit/test_schema_interfaces.py`
- `schema_interface_coverage_matrix.md`
- `schema_test_migration_record.md`

数据包通过测试仓库根目录导入，统一测试入口使用包导入，不使用相对文件系统路径。例如：

```python
from tests.mesh.data.schema_cases import ALL_CASES
```

`tests/mesh/data/schema_cases/` 应包含 `__init__.py`；必要时在测试配置中确保仓库根目录位于 `sys.path`。

以上代码和记录资产在执行阶段建立；本次只完成任务设计资产。

## 执行入口

1. 先依据 Workflow 完成现有 Schema 接口和按形状测试盘点。
2. 以 `kb/design/mesh/mesh_module_contract.md` 及其引用的实体接口约定为测试数据构造的规范来源，冻结标准接口输出形状、能力矩阵和节点顺序测试要求；不得从当前代码或现有测试反推规范。
3. 建立按几何族拆分的跨形状 case 包与唯一接口测试文件。
4. 完成旧测试对照、定向 pytest、相关回归测试和维护演练。
5. 将生产缺陷、契约歧义和覆盖缺口分别回流到对应承接任务。
