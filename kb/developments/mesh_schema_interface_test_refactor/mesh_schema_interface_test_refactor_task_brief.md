# FEALPy | Task Brief | 新网格 Schema 标准接口测试重构

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/mesh_schema_interface_test_refactor/mesh_schema_interface_test_refactor_task_brief.md`
- **启用条件**：当需要形成本测试重构任务的稳定定位、输入输出和对象级完成判据时，本文件必须启用
- **适用范围**：FEALPy 新网格 Schema 标准接口测试重构

## 一、Task 定位

本任务是一个独立的测试工程化任务，承接当前新网格模块测试中“按形状分文件”造成的重复和契约覆盖分散问题。任务不负责修复 Schema 生产代码，而是建立跨形状、按标准接口组织的长期测试结构。

## 二、主目标

形成一套由统一测试数据驱动、按 Schema 标准接口参数化执行的测试体系。所有 Schema 共享同一套接口测试入口；不同形状通过能力声明和测试 case 进入适用的测试矩阵。

## 三、输入与输出预期

**输入：**

- [网格模块约定](../../design/mesh/mesh_module_contract.md)中规定的顶点和局部面顺序（强制）；
- `fealpy/mesh/schema/entity_schema.py` 及各 Schema 的[公开接口](../../design/mesh/mesh_entity_functions.md)；
- 当前 `tests/mesh/unit/schema/` 下按形状组织的测试；
- 项目 backend、pytest 和现有 Mesh 测试约定。

**输出：**

- `tests/mesh/data/schema_cases/`：按几何族拆分的跨形状测试数据和 case 定义；
- `tests/mesh/unit/test_schema_interfaces.py`：唯一 Schema 标准接口测试文件；
- 测试接口覆盖矩阵及迁移对照记录；
- 旧测试迁移、删除或保留的依据说明；
- 定向 pytest 和相关回归测试结果。

## 四、对象级完成判据

1. 测试数据和测试逻辑已分离，新增形状主要通过向对应数据模块接入 case，而不是复制测试函数。
2. 唯一 Schema 接口测试文件按标准接口分节，并通过参数化覆盖多种形状。
3. 基本几何、映射、Jacobian、法向/切向、形函数、积分、多重指标、拓扑元数据和 index 行为均有测试入口。
4. 至少包含规则与非规则四边形、规则与非规则六面体，并验证节点顺序相关不变量。
5. 旧测试的有效断言均已迁移、明确废弃或记录为不再适用。
6. 测试失败能够通过 case 名称和接口名称定位到具体问题。
7. 实际 pytest 结果可复现，且任务报告明确剩余失败和未覆盖边界。

## 五、后续设计接口

- 进入 Task Object Boundary，明确与生产实现、拓扑测试、mesher 测试和全量测试系统的边界。
- 进入 Task Target Asset Design，冻结测试数据、唯一测试文件、覆盖矩阵和迁移记录的命名。
- 进入 Task Verification and Validation Design，分别设计测试实现正确性和测试资产可维护性的证据。
- 进入 Task Execution Workflow，组织盘点、迁移、对照、删除和回归验证。

## 附录 A：版本演进记录

- **v0.1**：2026-07-21，AI Agent，建立任务简介。
