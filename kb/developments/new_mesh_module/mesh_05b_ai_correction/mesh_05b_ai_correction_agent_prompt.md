# FEALPy | Agent Prompt | mesh_05b_ai_correction

你是负责 FEALPy 新网格模块短周期修正的 AI Agent。请根据 2026-06-09 小组周例会 T7 / 行动项 A5，完成 `mesh_05b_ai_correction` 子任务。

## 必读材料

开始前必须阅读：

1. `kb/developments/new_mesh_module/mesh_05b_ai_correction/mesh_05b_ai_correction_task_brief.md`
2. `kb/developments/new_mesh_module/mesh_05_algorithm_migration/mesh_05_algorithm_migration_team_handoff.md`
3. `kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_module_interface.md`
4. `fealpy/mesh/schema/*.py`
5. `tests/mesh/unit/schema/*.py`

## 任务目标

请用最小必要改动完成以下目标：

1. 统一并测试 `multi_index` 策略：
   - 单纯形：edge / triangle / tetrahedron。
   - 张量单元：quadrilateral / hexahedron，支持一个阶数默认各向相同，也支持各方向不同阶数。
   - prism / pyramid 如未完整支持，记录限制。

2. 澄清并修正 `grad_lambda` 与形函数导数边界：
   - 不要把 `grad_lambda` 伪装成所有单元的通用形函数导数。
   - 单纯形保留 `grad_lambda`。
   - 张量单元如保留 `grad_lambda`，必须说明其语义限制；更高阶形函数导数另用明确命名。

3. 修正 `normal()` / `tangent()` 返回形状：
   - 优先统一为 `[实体数, 方向数, 几何维数]`。
   - 只要求 1D / 2D / 3D 常用情形。
   - `G == T` 时允许返回 `(N, 0, G)` 空方向张量。
   - 每个修正必须有 shape 断言测试。

4. 跑通基本简单算例或等价单测：
   - edge 基本几何与 multi_index。
   - triangle / quadrilateral 基本几何、`bc_to_point`、`grad_lambda` 或等价几何导数。
   - tetrahedron / hexahedron 至少一类 3D 基本几何与法向 / 切向形状验证。

5. 扫描旧类型依赖：
   - 识别 `from fealpy.mesh import TriangleMesh`、`TriangleMesh(...)`、`from fealpy.mesh_old ...` 等显式依赖。
   - 形成 `mesh_05b_ai_correction_old_type_dependency_report.md`。

## 执行纪律

- 先写失败测试，再改实现。
- 每次只修一个接口族或一个 schema。
- 不做四维以上泛化。
- 不重写架构，不新增大型抽象层。
- 不长期维护两套网格；兼容层只作为过渡方案记录。
- 所有测试命令与结果写入 `mesh_05b_ai_correction_execution_record.md`。

## 推荐测试命令

```bash
python -m pytest tests/mesh/unit/schema/test_edge_schema.py -q
python -m pytest tests/mesh/unit/schema/test_triangle_schema.py -q
python -m pytest tests/mesh/unit/schema/test_quadrilateral_schema.py -q
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
python -m pytest tests/mesh/unit/schema/test_hexahedron_schema.py -q
```

若 `pytest` 不可用，请记录环境阻塞，并写最小 Python 脚本直接导入 schema、构造 `MeshBlock` / `EntitySector` / `EntityContext` 做接口调用验证。

## 完成输出

至少输出：

1. 修改过的代码文件与测试文件清单。
2. 测试命令及真实结果。
3. 旧类型依赖扫描报告。
4. 剩余风险与待确认点。
