# FEALPy | Task Network Design | 新网格模块架构优化 任务网络设计

- **版本**：v0.2
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_03_optimize_architecture_task_network_design.md`
- **启用条件**：当需要为 AI Agent 制定线性执行计划时，本文件必须启用
- **适用范围**：新网格模块架构优化的线性执行序列与 AI 辅助编程工作流

本文档用于表达当前 Task 的线性执行结构，适合 AI Agent 逐步自主完成、人工按关键节点验收的工作模式。

## 一、结构设计目标

将 8 项架构优化拆分为**线性优先级链**，每一步都清晰定义输入、操作、输出与验收标准。支持 AI Agent 逐步自主完成，人工在关键检查点验收。这种方式符合 AI 辅助编程的最佳实践，避免复杂依赖关系，提高自动化程度。

## 二、9 步骤线性执行序列

```
执行顺序（从上到下严格按序）：

步骤 1  →  步骤 2/3 (可轻度并行)  →  步骤 4  →  步骤 5  →  步骤 6  →  步骤 7  →  步骤 8  →  步骤 9(人工)
└─缓存────└─元数据/几何─────────────────┘    └─邻接──┘    └─双重──┘    └─API──┘    └─文档─────────────┘
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    AI 自主完成部分（85-95%）                            人工验收部分（5-15%）
```

## 三、单步骤详细说明

### **步骤 1：缓存失效机制**（基础层，必须最先）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #4 |
| **目标** | 在 MeshBlock / BoundaryInferencer 中实现缓存清理逻辑 |
| **前置输入** | `fealpy/mesh/storage/mesh_storage.py` 中的 `_cache_boundary_info` 机制 |
| **核心操作** | <ul><li>在 `MeshBlock.add_sector()` 添加缓存清空</li><li>在 relation 写入路径清空缓存</li><li>或提供 `invalidate_boundary_cache()` 接口</li></ul> |
| **输出成果** | <ul><li>修改后的 `mesh_storage.py`</li><li>`tests/test_mesh_cache_invalidation.py`</li><li>缓存管理说明文档</li></ul> |
| **验收标准** | <ul><li>拓扑修改后重查边界，结果与变化同步</li><li>无缓存漏洞或重复计算</li><li>单元测试通过率 100%</li></ul> |
| **预期工时** | 2~3 天 |
| **AI Agent 操作清单** | <ul><li>locate: `fealpy/mesh/storage/mesh_storage.py` 的 `MeshBlock` 类</li><li>edit: `add_sector()` 方法，添加 `self._cache_boundary_info.clear()`</li><li>edit: relation 写入方法，添加缓存清空</li><li>create: `tests/test_mesh_cache_invalidation.py`，包含修改前后查询对比</li><li>verify: 运行测试，确保通过</li></ul> |
| **关键代码位置** | `fealpy/mesh/storage/mesh_storage.py` - MeshBlock 类 |

**人工检查点 CP1**：缓存机制实现完毕，单测通过

---

### **步骤 2：opposite-vertex 元数据补充**（可与步骤 3 轻度并行）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #3 |
| **目标** | 为所有 simplex schema 补充 opposite-vertex 映射与方向约定 |
| **前置输入** | `fealpy/mesh/schema/` 中的 TriangleSchema、EdgeSchema 等 |
| **核心操作** | <ul><li>在 TriangleSchema 中添加 `opposite_vertex_edges` 属性</li><li>记录"第 i 条边与哪个顶点相对"</li><li>在代码注释中说明局部边顺序约定</li><li>为 Line / Tetrahedron 等其他 schema 补充类似说明</li></ul> |
| **输出成果** | <ul><li>修改后的 `schema/*.py` 文件</li><li>`kb/design/mesh/mesh_schema_metadata_reference.md`（元数据参考表）</li></ul> |
| **验收标准** | <ul><li>所有 simplex schema 都有明确的元数据或文档</li><li>代码中可查询到 opposite-vertex 映射</li><li>文档清晰可读</li></ul> |
| **预期工时** | 2~3 天 |
| **AI Agent 操作清单** | <ul><li>read: `fealpy/mesh/schema/` 下所有 schema 文件</li><li>edit: TriangleSchema，添加 `opposite_vertex_edges` 属性</li><li>edit: 其他 schema，补充相应元数据或文档</li><li>create: `kb/design/mesh/mesh_schema_metadata_reference.md`</li><li>verify: 代码审查，检查元数据完整性</li></ul> |
| **关键代码位置** | `fealpy/mesh/schema/triangle.py`, `fealpy/mesh/schema/edge.py`, ... |

**人工检查点 CP2.1**：元数据补充完毕

---

### **步骤 3：几何量维数自适应**（可与步骤 2 轻度并行）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #2 |
| **目标** | 修改 measure() / normal() 等接口支持二维/三维自动适配 |
| **前置输入** | `fealpy/mesh/schema/` 中的几何接口实现 |
| **核心操作** | <ul><li>检测坐标维数：`GD = positions.shape[1]`</li><li>二维：使用行列式面积公式 `abs(v1[0]*v2[1] - v1[1]*v2[0]) / 2`</li><li>三维：使用叉乘范数 `np.linalg.norm(np.cross(v1, v2)) / 2`</li><li>维数不匹配时抛出清晰的 ValueError 或自动投影</li></ul> |
| **输出成果** | <ul><li>修改后的 `schema/triangle.py`, `schema/tetrahedron.py`</li><li>`tests/test_mesh_geometric_interface.py`（二维/三维对比测试）</li></ul> |
| **验收标准** | <ul><li>二维网格能正确计算面积</li><li>三维网格能正确计算体积</li><li>维数不匹配时有清晰错误提示</li><li>结果与解析参考一致</li></ul> |
| **预期工时** | 3~4 天 |
| **AI Agent 操作清单** | <ul><li>locate: `TriangleSchema.measure()` 和 `normal()` 方法</li><li>edit: 添加维数检查，二维/三维分支处理</li><li>edit: TetrahedronSchema 等相应调整</li><li>create: `tests/test_mesh_geometric_interface.py`，使用标准网格（正方形、正四面体）验证</li><li>verify: 测试二维 box 网格与三维 tetra 网格</li></ul> |
| **关键代码位置** | `fealpy/mesh/schema/triangle.py`, `fealpy/mesh/schema/tetrahedron.py` |

**人工检查点 CP2.2**：几何接口维数自适应完毕，测试通过

---

### **步骤 4：EntitySchema 接口管理完善**

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #5 |
| **目标** | 区分已实现与预留接口，改进调用时的错误提示 |
| **前置输入** | `fealpy/mesh/schema/base.py` 中的 EntitySchema 基类 |
| **核心操作** | <ul><li>在 EntitySchema 中添加 `get_capabilities()` 方法</li><li>各 schema 实现该方法，返回支持的接口列表</li><li>EntityView 在调用可能未实现的接口前检查能力</li><li>改进错误信息，从 `AttributeError` 变为 `"TriangleSchema 不支持该操作"`</li></ul> |
| **输出成果** | <ul><li>修改后的 `schema/base.py` 与各 schema 类</li><li>修改后的 `view/entity_view.py`</li><li>`kb/design/mesh/mesh_schema_capabilities.md`（能力清单）</li></ul> |
| **验收标准** | <ul><li>EntitySchema 有 `get_capabilities()` 接口</li><li>调用未实现接口时错误提示清晰</li><li>能力清单文档准确完整</li></ul> |
| **预期工时** | 3~4 天 |
| **AI Agent 操作清单** | <ul><li>edit: `fealpy/mesh/schema/base.py`，添加 `get_capabilities()` 抽象方法</li><li>edit: 各 schema 类，实现 `get_capabilities()` 方法</li><li>edit: `fealpy/mesh/view/entity_view.py`，在调用前检查能力</li><li>create: `kb/design/mesh/mesh_schema_capabilities.md`</li><li>verify: 单元测试检查能力声明和错误提示</li></ul> |
| **关键代码位置** | `fealpy/mesh/schema/base.py`, `fealpy/mesh/view/entity_view.py` |

**人工检查点 CP3**：接口管理机制完毕，错误提示清晰

---

### **步骤 5：反向关系推断与邻接查询**（依赖步骤 1）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #1 |
| **目标** | 提供反向关系推断与邻接查询接口 |
| **前置依赖** | 步骤 1 完成（缓存一致性） |
| **前置输入** | `fealpy/mesh/topology/inferer.py` 与 `fealpy/mesh/view/` |
| **核心操作** | <ul><li>扩展 `TopologyInferer.infer()` 支持反向推断</li><li>或在 `Mesh` / `EntityView` 中添加 `reverse_relation()` 方法</li><li>实现 `edge.to_reverse()` 或 `mesh.reverse_relation("tri", "edge")`</li><li>保证反向关系与缓存机制兼容</li></ul> |
| **输出成果** | <ul><li>修改后的 `topology/inferer.py` 或 `view/*.py`</li><li>`tests/test_mesh_relation_inference.py`</li></ul> |
| **验收标准** | <ul><li>用户能直接查询 `edge → tri` 邻接</li><li>结果与前向关系对称一致</li><li>边界边邻接数为 1，内部边为 2</li></ul> |
| **预期工时** | 4~5 天 |
| **AI Agent 操作清单** | <ul><li>design: 反向关系存储与推断机制</li><li>edit: `topology/inferer.py` 添加反向推断逻辑</li><li>或 edit: `view/mesh.py` / `entity_view.py` 添加便捷接口</li><li>create: `tests/test_mesh_relation_inference.py`</li><li>verify: 用 mesh_01_validation 中的标准网格验证</li></ul> |
| **关键代码位置** | `fealpy/mesh/topology/inferer.py` 或 `fealpy/mesh/view/mesh.py` |

**人工检查点 CP4**：反向邻接查询功能工作正常

---

### **步骤 6：形状+维数双重实体选取**（依赖步骤 5）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #7 |
| **目标** | 支持按形状名称+拓扑维数组合查询实体 |
| **前置依赖** | 步骤 5 完成（可靠的实体关系） |
| **前置输入** | `fealpy/mesh/view/mesh.py` 中的 `sector()` 方法 |
| **核心操作** | <ul><li>在 Mesh 中添加 `entity(shape, dimension)` 方法</li><li>或扩展 `sector()` 支持维数参数：`sector("triangle", dimension=2)`</li><li>添加维数验证与错误提示</li><li>示例：`mesh.entity("triangle", 2)` 返回二维三角形</li></ul> |
| **输出成果** | <ul><li>修改后的 `view/mesh.py`</li><li>使用示例与文档</li></ul> |
| **验收标准** | <ul><li>API 形式直观简洁</li><li>维数不匹配时有清晰提示</li><li>查询结果正确</li></ul> |
| **预期工时** | 2~3 天 |
| **AI Agent 操作清单** | <ul><li>edit: `fealpy/mesh/view/mesh.py`，添加 `entity(shape, dimension)` 或扩展 `sector()`</li><li>add: 维数验证逻辑</li><li>write: 使用示例与 docstring</li><li>verify: 简单集成测试</li></ul> |
| **关键代码位置** | `fealpy/mesh/view/mesh.py` |

**人工检查点 CP5**：双重实体查询接口就绪

---

### **步骤 7：view 层便利查询 API**（依赖步骤 5、6）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #8 |
| **目标** | 在 EntityView / Mesh 中增加常用的邻接、边界、梯度查询接口 |
| **前置依赖** | 步骤 5（反向邻接）、步骤 6（双重查询） |
| **前置输入** | `fealpy/mesh/view/entity_view.py` 与 `mesh.py` |
| **核心操作** | <ul><li>实现 `cell_to_cell_neighbors(cell_type)` - 邻接单元</li><li>实现 `on_boundary(entity_type)` - 边界实体</li><li>实现 `gradient_indices(entity_type)` - 梯度计算索引</li><li>（可选）`parent_entity()`, `child_entities()` 等</li></ul> |
| **输出成果** | <ul><li>修改后的 `view/entity_view.py` / `mesh.py`</li><li>`tests/test_mesh_view_api.py`</li><li>API 文档与使用示例</li></ul> |
| **验收标准** | <ul><li>3+ 个便利 API 完整实现</li><li>能在 mesh_01_validation 算例中直接使用</li><li>测试覆盖完整</li></ul> |
| **预期工时** | 4~5 天 |
| **AI Agent 操作清单** | <ul><li>design: 便利 API 的签名与返回值形式</li><li>implement: 3+ 个 API，基于步骤 5 的反向关系与步骤 1 的缓存机制</li><li>create: `tests/test_mesh_view_api.py`，单元测试</li><li>write: 详细文档与使用示例</li><li>integrate: 改写 mesh_01_validation 算例展示新 API</li></ul> |
| **关键代码位置** | `fealpy/mesh/view/entity_view.py`, `fealpy/mesh/view/mesh.py` |

**人工检查点 CP6**：所有便利 API 实现完毕，算例可用

---

### **步骤 8：文档与术语统一**（依赖所有前置步骤）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 优化项 #6 |
| **目标** | 同步设计文档与代码术语，生成迁移指南 |
| **前置依赖** | 步骤 1~7 全部完成 |
| **前置输入** | `kb/design/mesh/` 中的设计文档 + 步骤 1~7 的代码改动 |
| **核心操作** | <ul><li>更新设计文档为代码术语（MeshBlock 而非 MeshStorage）</li><li>生成术语映射表（设计术语 ↔ 代码术语）</li><li>编写迁移指南（如何用新 API 改写现有代码）</li><li>编写 API 变更总结文档</li><li>编写优化前后的对比分析</li></ul> |
| **输出成果** | <ul><li>更新的 `kb/design/mesh/mesh-data-structure.md` 等</li><li>`kb/design/mesh/mesh_terminology_mapping.md`</li><li>`kb/explanation/mesh_optimization_migration_guide.md`</li><li>`mesh_03_optimize_architecture_before_after_comparison.md`</li></ul> |
| **验收标准** | <ul><li>设计文档与代码术语完全一致</li><li>术语映射表清晰准确</li><li>迁移指南可指导用户改写代码</li></ul> |
| **预期工时** | 3~4 天 |
| **AI Agent 操作清单** | <ul><li>read: 所有前置步骤的代码与文档</li><li>update: `kb/design/mesh/` 中的所有文档，统一术语</li><li>create: `mesh_terminology_mapping.md`（表格形式）</li><li>create: `kb/explanation/mesh_optimization_migration_guide.md`</li><li>create: 优化前后对比分析文档</li><li>verify: 文档审查，确保术语一致</li></ul> |
| **关键文档位置** | `kb/design/mesh/`, `kb/explanation/` |

**人工检查点 CP7**：文档与术语同步完毕，迁移指南可用

---

### **步骤 9：集成验证与回归测试**（人工验收）

| 属性 | 值 |
|-----|-----|
| **优化映射** | 综合验收 |
| **目标** | 验证所有优化的集成效果与向后兼容性 |
| **前置依赖** | 步骤 1~8 全部完成 |
| **核心活动** | <ul><li>运行 mesh_01_validation 中的 Line Walk 与四面体质量算例</li><li>验证现有测试套件通过率</li><li>统计代码覆盖率</li><li>人工审查主要改动点</li><li>签署交付确认</li></ul> |
| **输出成果** | <ul><li>集成验证报告</li><li>测试覆盖率报告</li><li>最终交付清单与签署</li></ul> |
| **验收标准** | <ul><li>mesh_01_validation 算例完整运行</li><li>测试覆盖率 ≥ 80%</li><li>无回归，现有测试 100% 通过</li><li>无新的 lint 错误</li></ul> |
| **工时** | 1~2 天（人工为主） |
| **验收清单** | <ul><li>✅ Line Walk 算例运行通过，反向邻接 API 有效</li><li>✅ 四面体质量算例运行通过，几何 API 可用</li><li>✅ pytest 整体通过率 100%</li><li>✅ 新增代码覆盖率 ≥ 80%</li><li>✅ pylint / flake8 无新增错误</li><li>✅ 设计文档与代码术语一致</li><li>✅ 迁移指南清晰可用</li><li>✅ 所有 git 提交历史清晰</li></ul> |

**最终交付**：Task 正式完成，所有成果入库

---

## 四、AI Agent 执行时间线

```
预期总工时：5~6 周（AI 自主完成 85-95%）

[第 1 周]
  步骤 1（缓存）        │ ████ │ 2~3 天
                        │      ├─→ CP1：缓存机制完毕

[第 1~2 周]
  步骤 2（元数据）      │ ████ │ 2~3 天
  + 步骤 3（几何）      │ █████ │ 3~4 天 [轻度并行]
                        │      ├─→ CP2.1 / CP2.2：两项基础工作完毕

[第 2 周]
  步骤 4（接口管理）    │ █████ │ 3~4 天
                        │      ├─→ CP3：接口管理完毕

[第 3 周]
  步骤 5（反向邻接）    │ ██████│ 4~5 天（依赖步骤 1）
                        │      ├─→ CP4：邻接查询工作正常

[第 3.5~4 周]
  步骤 6（双重查询）    │ ████ │ 2~3 天（依赖步骤 5）
                        │      ├─→ CP5：双重查询就绪

[第 4~4.5 周]
  步骤 7（便利 API）    │ ██████│ 4~5 天（依赖步骤 5、6）
                        │      ├─→ CP6：所有 API 完毕

[第 5 周]
  步骤 8（文档）        │ █████ │ 3~4 天（依赖步骤 1~7）
                        │      ├─→ CP7：文档与术语同步

[第 5.5 周]
  AI 交付准备           │ ██   │ 1~2 天
  整理成果、生成清单

[第 6 周]
  步骤 9（人工验收）    │ █████ │ 1~2 天
  ✅ 最终交付
```

## 五、人工验收检查点总览

| CP | 时机 | 检查项 | 通过标准 | 如有问题 |
|---|------|--------|---------|---------|
| **CP1** | 步骤 1 后 | 缓存失效机制 | 单测通过、无缓存漏洞 | 重做步骤 1 |
| **CP2.1** | 步骤 2 后 | 元数据补充 | 所有 schema 有明确元数据 | 补充缺失项 |
| **CP2.2** | 步骤 3 后 | 几何接口 | 二维/三维都能计算 | 调整维数检查 |
| **CP3** | 步骤 4 后 | 接口管理 | 错误提示清晰 | 改进错误信息 |
| **CP4** | 步骤 5 后 | 反向邻接 | Line Walk 可用 | 重设计推断逻辑 |
| **CP5** | 步骤 6 后 | 双重查询 | API 直观可用 | 调整 API 形式 |
| **CP6** | 步骤 7 后 | 便利 API | 3+ API 在算例中可用 | 补充或重做 |
| **CP7** | 步骤 8 后 | 文档术语 | 设计文档 = 代码术语 | 更新文档 |
| **CP8** | 步骤 9 后 | 集成验收 | mesh_01_validation 通过 | 修复回归 |

## 六、失败恢复策略

- **步骤 1 失败**：影响步骤 5、7 的缓存安全性；需重做步骤 1 及其依赖项（5、7）
- **步骤 2、3 失败**：独立问题，可单独重做
- **步骤 4 失败**：仅影响接口管理，可独立修复
- **步骤 5 失败**：影响步骤 6、7；需重设计推断逻辑，重做 5、6、7
- **步骤 6 失败**：仅影响该步骤，可单独重做
- **步骤 7 失败**：可单独重做或增补 API
- **步骤 8 失败**：文档重做，不影响代码
- **步骤 9 失败**：根据具体问题定向修复相应步骤

**关键原则**：git 提交历史清晰，每步完成后独立提交，便于追踪与回滚。

## 七、工作量与资源统计

| 环节 | 预期工时 | AI 参与度 | 人工参与度 | 备注 |
|------|---------|---------|----------|------|
| 步骤 1~8（开发）| 5~6 周 | 95% | 5%（检查点） | AI 自主完成 |
| 步骤 9（验收）| 1~2 周 | 5% | 95% | 人工为主 |
| **总计** | **6~7 周** | **85%** | **15%** | 中等规模 AI 辅助任务 |

## 八、AI Agent 最佳实践建议

1. **每步独立 git 提交**：步骤完成后立即 `git commit`，便于追踪与回滚
2. **清晰的单元测试**：每步添加对应的单元测试，不等待集成验收
3. **代码审查友好**：改动清晰、注释充分，便于人工检查点的快速验收
4. **文档同步**：代码改动时同时更新 docstring 和注释，避免后期补救
5. **增量构建**：不跳步，不并行相关步骤，保证依赖关系清晰
6. **验收材料准备**：每步准备好验收清单，便于 CP 检查

## 附录 A：本文件版本演进记录

- **v0.2**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 将任务网络从并行模式改为线性链式结构
		- 设计适合 AI Agent 逐步完成的 9 步骤序列
		- 明确每个步骤的输入、操作、输出、验收标准与工时
		- 添加人工检查点与失败恢复策略
		- 优化为适合 AI 辅助编程的最佳实践

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 首次建立新网格模块架构优化 Task Network Design
		- 设计 9 节点的并行与串联结构
