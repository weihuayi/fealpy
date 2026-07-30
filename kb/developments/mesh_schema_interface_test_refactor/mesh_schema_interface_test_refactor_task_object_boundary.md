# FEALPy | Task Object Boundary | 新网格 Schema 标准接口测试重构 对象边界

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/mesh_schema_interface_test_refactor/mesh_schema_interface_test_refactor_task_object_boundary.md`
- **启用条件**：当需要界定本测试重构任务直接建设什么、不直接建设什么以及相邻任务分界时，本文件必须启用
- **适用范围**：新网格 Schema 标准接口测试的组织重构、数据资产和回归迁移

## 一、直接建设对象

测试数据的规范基线是 [`kb/design/mesh/mesh_module_contract.md`](../../design/mesh/mesh_module_contract.md) 及其引用的接口约定；代码现状不是测试数据契约的来源。

本任务直接建设：

- `tests/mesh/data/schema_cases/` 下按几何族拆分的跨形状 Schema 测试 case、统一 context builder 和参考值/参考计算；
- `tests/mesh/unit/test_schema_interfaces.py` 唯一标准接口测试入口；
- Schema 接口能力矩阵、测试 case 命名和参数化约定；
- 旧测试到新测试的覆盖迁移对照；
- 与规范中定义的节点排序、局部面顺序、映射、Jacobian、法向、测度、积分和 index 契约相关的断言；
- 测试迁移后的执行说明和失败定位信息。

## 二、不直接建设对象

本任务不直接建设：

- `fealpy/mesh/schema/` 的生产实现和接口语义变更；
- 新形状、新算法或新 Mesh API；
- 全部 Mesh view、TopologyBuilder、mesher、IO 和兼容层测试的重构；
- CI 平台、pytest 插件或覆盖率基础设施改造；
- 因测试暴露出的生产 Bug 的修复；
- 最终发布验收或新网格模块总体可用性结论。

## 三、相邻任务分界

- **与 `mesh_07_module_validation` 的关系**：参考其问题发现和回归需求，但不以其观察到的代码行为作为测试契约；节点顺序和局部面顺序以 [`kb/design/mesh/mesh_module_contract.md`](../../design/mesh/mesh_module_contract.md) 为准，将规范转化为测试数据和断言；不重复进行上层研究验证。
- **与生产实现任务的关系**：测试失败只提供可复现证据和回流入口，不在本任务内修改生产实现。
- **与拓扑/mesher 测试的关系**：Schema 标准接口测试验证接口可观察行为；拓扑构造器和网格生成器的专项内部行为仍由各自测试资产负责。
- **与 `mesh_02_test_system` 的关系**：复用其测试执行和可重复性原则；本任务不重新建设通用测试系统。
- **与后续覆盖率或验收任务的关系**：输出测试矩阵和缺口清单，供后续任务承接，不替代其覆盖率结论或最终验收。

## 四、边界判定规则

1. 只要测试对象是 `EntitySchema` 的标准公开接口，可进入统一测试文件。
2. 如果测试依赖某个具体 Mesh 工厂、文件格式或上层求解流程，则不自动进入统一 Schema 文件。
3. 如果不同形状的数学性质相同，应通过 case 参数化共用测试；如果性质确实不同，应在能力声明或参考计算中表达差异，而不是复制整套文件。
4. 发现生产缺陷时保留失败测试和最小复现，创建回流项后再决定是否暂时标记 xfail；不得删除失败以获得绿色结果。

## 五、待核验边界

- 是否需要将 `global_permutations()` 的全部自由度排列测试纳入统一文件，还是仅保留契约级测试并将详细排列放到专项测试。
- 旧测试文件删除时，哪些拓扑断言应迁移到现有 topology 测试文件。

## 附录 A：版本演进记录

- **v0.1**：2026-07-21，AI Agent，建立任务对象边界。
