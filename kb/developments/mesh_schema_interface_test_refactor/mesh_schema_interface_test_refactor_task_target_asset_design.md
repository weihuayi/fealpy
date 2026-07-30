# FEALPy | Task Target Asset Design | 新网格 Schema 标准接口测试重构 目标资产设计

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/mesh_schema_interface_test_refactor/mesh_schema_interface_test_refactor_task_target_asset_design.md`
- **启用条件**：当本任务需要统一设计测试数据、测试入口、覆盖矩阵和迁移记录时，本文件必须启用
- **适用范围**：Schema 标准接口测试重构目标资产

## 一、目标资产集合定位

本任务的目标不是增加零散测试，而是建立一组彼此配合的长期测试资产，使不同 Schema 通过同一标准接口测试入口接受验证。

## 二、目标资产清单

| 编号   | 目标资产             | 类型                          | 建议位置                                                                                      | 主位职责                             |
| ---- | ---------------- | --------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------- |
| A-01 | 跨形状 Schema 测试数据包 | Test Data Asset             | `tests/mesh/data/schema_cases/`                                                           | 按几何族拆分 case、共享 context、能力声明和参考数据 |
| A-02 | Schema 标准接口测试入口  | Test Asset                  | `tests/mesh/unit/test_schema_interfaces.py`                                               | 按接口组织参数化测试                       |
| A-03 | Schema 接口覆盖矩阵    | Knowledge / Test Design     | `kb/developments/mesh_schema_interface_test_refactor/schema_interface_coverage_matrix.md` | 记录接口、形状、case、规范依据、断言和状态          |
| A-04 | 测试迁移对照记录         | Knowledge / Migration Asset | `kb/developments/mesh_schema_interface_test_refactor/schema_test_migration_record.md`     | 追踪旧断言迁移、废弃和遗留失败，并记录代码现状与规范不一致    |
| A-05 | 独立参考计算辅助         | Test Support Asset          | `tests/mesh/data/schema_cases/references.py`                                              | 为非仿射积分等提供独立参考计算                  |

## 三、资产关系

```text
Schema 接口定义 + 旧形状测试
              |
              v
      A-01 测试数据与 case
              |
              +--> A-02 唯一接口测试入口
              |
              +--> A-03 接口覆盖矩阵
              |
              +--> A-04 迁移对照记录
              |
              +--> A-05 独立参考计算
```

## 四、质量关注点

- case 名称能定位几何情形、Schema 和 backend；
- case 的顶点、局部面和关系数据可追溯到 `kb/design/mesh/mesh_module_contract.md`，不从生产代码现状复制；
- 测试数据不依赖外部文件和全局随机状态；
- 测试逻辑不复制形状专属模板；
- 标准输出形状、有限性、index 和跨接口关系有显式断言；
- 非仿射参考计算不调用被测生产积分实现；
- 旧测试有效覆盖没有静默丢失；
- 测试失败信息包含接口名和 case 名；
- 新增形状能够通过新增数据和能力声明接入。
- 新增形状能够通过新增或修改对应几何族数据模块和能力声明接入，不需要修改所有接口测试函数。
- 数据模块按几何族划分，降低多人协作时的修改冲突；接口测试入口仍保持唯一。

## 六、数据包组织与导入约定

建议的数据包结构为：

```text
tests/mesh/data/schema_cases/
├── __init__.py          # 汇总并导出 ALL_CASES 及按能力筛选的集合
├── common.py            # SchemaCase、context builder、通用测试输入
├── simplex.py           # point、segment、triangle、tetrahedron
├── tensor_product.py    # quadrilateral、hexahedron
├── mixed.py             # prism、pyramid
└── references.py        # 独立解析参考值和 Jacobian 测度计算
```

这里按几何族拆分数据，不按接口拆分数据，也不把每个“形状 × 接口”组合复制一份。每个 `SchemaCase` 声明 `supported_interfaces`，`test_schema_interfaces.py` 根据该声明筛选参数。

`tests/mesh/unit/test_schema_interfaces.py` 与 `tests/mesh/data/schema_cases/` 虽然位于不同子目录，但测试运行时应以仓库根目录作为导入根，通过包导入：

```python
from tests.mesh.data.schema_cases import ALL_CASES
```

不得用 `Path(__file__).parents[...]` 拼接数据文件路径，也不得用 `sys.path` 指向 `tests/mesh/data` 后导入裸模块。若当前测试布局缺少包标记，应添加最小的 `__init__.py`，不应改变生产包安装配置。

## 五、验收接口

A-01/A-05 为测试入口提供输入，A-02 产生 pytest 结果，A-03/A-04 提供覆盖可追溯性。测试结果可回流到生产实现任务；缺口清单可交给测试系统或最终验收任务。

## 附录 A：版本演进记录

- **v0.1**：2026-07-21，AI Agent，建立目标资产设计。
