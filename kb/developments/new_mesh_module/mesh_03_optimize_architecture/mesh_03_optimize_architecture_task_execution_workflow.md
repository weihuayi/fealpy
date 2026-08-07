# FEALPy | Task Execution Workflow | 新网格模块架构优化 执行工作流

- **版本**：v0.2
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_03_optimize_architecture_task_execution_workflow.md`
- **启用条件**：当需要定义当前 Task 的 AI Agent 友好的线性执行计划时，本文件必须启用
- **适用范围**：新网格模块架构优化的线性执行工作流与 AI 辅助编程最佳实践

本文档用于定义当前 Task 的执行 How，支持 AI Agent 逐步自主完成、人工在关键点验收的工作模式。

## 一、Workflow 定位

本 Workflow 将 Task Network Design 中的 9 步骤映射到具体的执行活动，支持 AI Agent 逐步完成、人工按检查点验收。整个任务预计 6~7 周完成，其中 AI 自主完成 85-95%。

## 二、总体执行模式

**执行模式**：线性优先级链 + 人工检查点 + AI 自主完成
- 9 个步骤严格按序进行（部分可轻度并行）
- 每步完成后有一个人工检查点（CP1~CP8）
- 第 9 步（步骤 9）是人工为主的综合验收
- 总工时：6~7 周，AI 参与 85%，人工参与 15%

## 三、逐周执行计划概览

| 周次      | 步骤编号  | 步骤名称   | 工时   | 关键检查点               | 依赖    |
| ------- | ----- | ------ | ---- | ------------------- | ----- |
| W1      | S1    | 缓存失效机制 | 2-3d | CP1 (30min)         | 无     |
| W1~W2   | S2+S3 | 元数据+几何 | 2-4d | CP2.1+CP2.2 (50min) | 无     |
| W2      | S4    | 接口管理   | 3-4d | CP3 (25min)         | 无     |
| W3      | S5    | 反向邻接   | 4-5d | CP4 (35min)         | S1    |
| W3.5~W4 | S6    | 双重查询   | 2-3d | CP5 (15min)         | S5    |
| W4~W4.5 | S7    | 便利 API | 4-5d | CP6 (40min)         | S5+S6 |
| W5      | S8    | 文档术语   | 3-4d | CP7 (30min)         | S1~S7 |
| W5.5    | -     | AI 准备  | 1-2d | -                   | S1~S8 |
| W6      | S9    | 集成验收   | 1-2d | CP8 (最终)            | S1~S8 |

## 四、详细步骤执行清单

### 步骤 1（W1）：缓存失效机制

**执行活动**：
1. 分析 `fealpy/mesh/storage/mesh_storage.py` 的 MeshBlock 缓存机制
2. 在 `add_sector()` 添加缓存清空逻辑
3. 在 relation 写入路径添加缓存清空
4. 编写单元测试 `tests/test_mesh_cache_invalidation.py`
5. 验证所有测试通过

**AI 操作清单**：
- [ ] locate: MeshBlock class in mesh_storage.py
- [ ] edit: add_sector() method - add cache invalidation
- [ ] edit: relation write paths - add invalidation calls
- [ ] create: tests/test_mesh_cache_invalidation.py
- [ ] run: pytest tests/test_mesh_cache_invalidation.py -v
- [ ] verify: all tests pass (100%)
- [ ] commit: "feat(mesh): implement cache invalidation mechanism"

**人工检查点 CP1**（30 分钟）：
- ✅ 缓存机制是否正确实现
- ✅ 单元测试通过 100%
- ✅ 无缓存漏洞（修改后重查边界结果一致）

---

### 步骤 2（W1~W2）：opposite-vertex 元数据补充

**执行活动**：
1. 读取 `fealpy/mesh/schema/` 下所有 schema 文件
2. 为 TriangleSchema 添加 `opposite_vertex_edges` 属性
3. 为其他 schema 补充相应元数据或文档
4. 生成 `kb/design/mesh/mesh_schema_metadata_reference.md`

**AI 操作清单**：
- [ ] read: all files in fealpy/mesh/schema/
- [ ] edit: TriangleSchema - add opposite_vertex_edges
- [ ] edit: EdgeSchema, LineSchema, etc - add metadata
- [ ] create: kb/design/mesh/mesh_schema_metadata_reference.md
- [ ] verify: code review for completeness
- [ ] commit: "docs(mesh): add schema metadata and opposite-vertex mapping"

**人工检查点 CP2.1**（20 分钟）：
- ✅ 所有 schema 都有明确的元数据
- ✅ opposite-vertex 映射正确
- ✅ 文档清晰可读

---

### 步骤 3（W1~W2）：几何量维数自适应

**执行活动**：
1. 修改 `TriangleSchema.measure()` - 维数检测与分支处理
2. 修改 `normal()` 方法 - 支持二维语义
3. 修改 TetrahedronSchema 等相应接口
4. 编写单元测试 `tests/test_mesh_geometric_interface.py`
5. 验证二维和三维计算结果

**AI 操作清单**：
- [ ] locate: TriangleSchema.measure() and normal()
- [ ] edit: add dimension detection and 2D/3D branches
- [ ] edit: TetrahedronSchema and other schemas
- [ ] create: tests/test_mesh_geometric_interface.py
- [ ] run: pytest tests/test_mesh_geometric_interface.py -v
- [ ] verify: 2D and 3D calculations correct
- [ ] commit: "feat(mesh): add dimension-adaptive geometric interfaces"

**人工检查点 CP2.2**（30 分钟）：
- ✅ 二维网格能正确计算面积
- ✅ 三维网格能正确计算体积
- ✅ 维数不匹配有清晰提示
- ✅ 单元测试通过

---

### 步骤 4（W2）：EntitySchema 接口管理

**执行活动**：
1. 在 EntitySchema 基类添加 `get_capabilities()` 方法
2. 各具体 schema 实现该方法
3. 修改 EntityView 在调用前检查能力
4. 改进错误提示信息
5. 生成能力清单文档

**AI 操作清单**：
- [ ] edit: fealpy/mesh/schema/base.py - add get_capabilities()
- [ ] edit: all schema classes - implement get_capabilities()
- [ ] edit: fealpy/mesh/view/entity_view.py - add capability check
- [ ] improve: error messages for unsupported operations
- [ ] create: kb/design/mesh/mesh_schema_capabilities.md
- [ ] verify: unit tests for capability declaration
- [ ] commit: "feat(mesh): add schema capability management"

**人工检查点 CP3**（25 分钟）：
- ✅ EntitySchema 有 get_capabilities() 接口
- ✅ 错误提示清晰有用
- ✅ 能力清单准确完整

---

### 步骤 5（W3）：反向关系推断与邻接查询

**执行活动**：
1. 设计反向关系推断机制
2. 扩展 TopologyInferer.infer() 或在 Mesh 层新增接口
3. 实现 reverse_relation() 或 to_reverse() 方法
4. 编写单元测试 `tests/test_mesh_relation_inference.py`
5. 验证对称性和边界条件

**AI 操作清单**：
- [ ] design: reverse relation inference mechanism
- [ ] edit: fealpy/mesh/topology/inferer.py or view/mesh.py
- [ ] implement: reverse_relation() or similar API
- [ ] create: tests/test_mesh_relation_inference.py
- [ ] verify: tri→edge ↔ edge→tri symmetry
- [ ] verify: boundary edges have 1 neighbor, interior have 2
- [ ] commit: "feat(mesh): implement reverse relation inference"

**人工检查点 CP4**（35 分钟）：
- ✅ 反向邻接查询工作正常
- ✅ Line Walk 算例可用
- ✅ 对称性与边界检查通过
- ✅ 单元测试通过

**依赖**：步骤 1（缓存一致性）

---

### 步骤 6（W3.5~W4）：双重实体选取

**执行活动**：
1. 在 Mesh 类添加 `entity(shape, dimension)` 方法
2. 实现维数验证与错误提示
3. 编写 docstring 和使用示例
4. 简单的集成测试

**AI 操作清单**：
- [ ] edit: fealpy/mesh/view/mesh.py - add entity() method
- [ ] add: dimension validation logic
- [ ] write: docstring with examples
- [ ] verify: basic integration test
- [ ] commit: "feat(mesh): add shape+dimension dual entity selection"

**人工检查点 CP5**（15 分钟）：
- ✅ API 形式直观简洁
- ✅ 维数验证逻辑正确
- ✅ 查询结果准确

**依赖**：步骤 5（可靠的关系）

---

### 步骤 7（W4~W4.5）：view 层便利 API

**执行活动**：
1. 设计便利 API 签名（cell_to_cell_neighbors, on_boundary, gradient_indices）
2. 实现 3+ 个便利 API
3. 编写单元测试 `tests/test_mesh_view_api.py`
4. 改写 mesh_01_validation 算例展示新 API
5. 补充文档和使用示例

**AI 操作清单**：
- [ ] design: API signatures and return types
- [ ] implement: cell_to_cell_neighbors()
- [ ] implement: on_boundary()
- [ ] implement: gradient_indices()
- [ ] create: tests/test_mesh_view_api.py
- [ ] rewrite: mesh_01_validation examples
- [ ] write: comprehensive documentation
- [ ] commit: "feat(mesh): add convenient query APIs"

**人工检查点 CP6**（40 分钟）：
- ✅ 3+ API 完整实现
- ✅ 在算例中可直接使用
- ✅ 测试覆盖完整
- ✅ 文档清晰

**依赖**：步骤 5（反向邻接）+ 步骤 6（双重查询）

---

### 步骤 8（W5）：文档与术语统一

**执行活动**：
1. 更新 `kb/design/mesh/` 所有设计文档，统一术语
2. 生成术语映射表
3. 生成用户迁移指南
4. 生成优化前后对比分析
5. 生成 API 变更总结

**AI 操作清单**：
- [ ] read: all design documents in kb/design/mesh/
- [ ] update: unify terminology to code terms
- [ ] create: mesh_terminology_mapping.md
- [ ] create: mesh_optimization_migration_guide.md
- [ ] create: before_after_comparison.md
- [ ] create: api_changes_summary.md
- [ ] verify: documentation review for consistency
- [ ] commit: "docs(mesh): unify terminology and create guides"

**人工检查点 CP7**（30 分钟）：
- ✅ 设计文档与代码术语完全一致
- ✅ 术语映射表清晰准确
- ✅ 迁移指南能指导用户改写代码
- ✅ 对比分析证明优化价值

**依赖**：步骤 1~7 全部完成

---

### 步骤 9（W6）：集成验证与回归测试

**验收活动**（人工为主）：
1. 运行 mesh_01_validation 的 Line Walk 算例
2. 运行四面体质量计算算例
3. 运行完整测试套件 `pytest`
4. 统计代码覆盖率
5. pylint / flake8 检查
6. 人工审查主要改动
7. 签署交付确认

**验收清单**：
- ✅ Line Walk 算例通过，反向邻接 API 有效
- ✅ 四面体质量算例通过，几何 API 可用
- ✅ pytest 通过率 100%
- ✅ 新增代码覆盖率 ≥ 80%
- ✅ 无新增 lint 错误
- ✅ 设计文档与代码术语一致
- ✅ 迁移指南清晰可用
- ✅ git 历史清晰

**人工检查点 CP8**（2~3 小时）：
完整的集成验收，所有检查点通过，现有功能无回归

---

## 五、AI Agent 执行最佳实践

### 代码提交纪律
```bash
# 每步完成后执行
git add -A
git commit -m "feat(mesh): <step description>"
```

### 单元测试编写
- 每个新功能对应一个测试文件
- 测试覆盖正常路径、边界情况、错误处理
- 使用标准网格（box, tetrahedron）作为测试数据

### 文档编写纪律
- docstring：参数、返回值、异常、使用示例
- 代码注释：解释 why，不是 what
- 术语一致：全程使用统一的术语

### 与人工检查点协作
- 提前准备验收材料：执行命令、预期结果、清单
- CP 反馈后立即修复，不积累问题
- 保持 git history 清晰，便于追踪

## 附录 A：检查点快速参考

```bash
# CP1：缓存失效
pytest tests/test_mesh_cache_invalidation.py -v

# CP2.1：元数据
grep -r "opposite_vertex" fealpy/mesh/schema/

# CP2.2：几何接口
pytest tests/test_mesh_geometric_interface.py -v

# CP3：接口管理
python -c "from fealpy.mesh.schema import TriangleSchema; print(TriangleSchema().get_capabilities())"

# CP4：反向邻接
pytest tests/test_mesh_relation_inference.py -v

# CP5：双重查询
python -c "from fealpy.mesh import Mesh; m = Mesh(...); e = m.entity('triangle', 2)"

# CP6：便利 API
pytest tests/test_mesh_view_api.py -v

# CP7：文档术语
diff kb/design/mesh/mesh-data-structure.md <(grep -o "MeshBlock\|EntitySector" fealpy/mesh/storage/*.py)

# CP8：集成验证
pytest && python examples/line_walk_new_mesh.py && python examples/tet_quality_new_mesh.py
```

## 附录 B：版本演进记录

- **v0.2**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 将执行流程从多轮并行改为线性顺序执行
		- 为每个步骤设计清晰的 AI 操作清单
		- 提供 8 个人工检查点（CP1~CP8）及快速检查方法
		- 完全适合 AI Agent 自主执行，人工按节点验收
		- 预计 6~7 周完成，AI 参与 85-95%

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 首次建立新网格模块架构优化的多阶段执行计划
		- 设计 4 个阶段 12 个 Node 的并行结构
