# N-01 / N-03 资产索引

任务：`mesh_schema_interface_test_refactor`

| 节点 | 文档 | 用途 |
|---|---|---|
| N-01 | `N-01_schema_contract_inventory.md` | 规范基线、形状/局部实体顺序、标准接口和 N-04 字段依据 |
| N-03 | `N-03_schema_interface_contract_freeze.md` | 可执行输出形状、断言规则、能力矩阵、失败分类和开放项 |

## 交接给 N-04

N-04 应先阅读 N-01，再按 N-03 的数据模型实现 `SchemaCase` 和按几何族拆分的数据包。优先顺序：

1. `common.py`：case 元数据、context builder、能力筛选。
2. `simplex.py`：point、segment、tri、tet。
3. `tensor_product.py`：quad、hex，必须包含规则/扭曲和顺序元数据。
4. `mixed.py`：prism、pyramid，按能力矩阵显式声明适用性。
5. `references.py`：独立的 Jacobian/测度/积分参考计算。
6. `__init__.py`：集中导出 `ALL_CASES`，不让 N-05 依赖几何族内部名称。

## 不应由 N-04 自行裁决

- 生产代码与规范冲突时的修复方案；
- 未在设计合同中给出的三维 orientation 全排列；
- prism/pyramid 未决接口是否强行加入统一测试；
- 通过当前实现反推期望值。

这些事项必须按 N-03 开放项或统一任务资产中的回流规则记录。
