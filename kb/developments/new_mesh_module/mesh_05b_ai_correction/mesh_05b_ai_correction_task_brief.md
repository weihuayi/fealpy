# FEALPy | Task Brief | mesh_05b_ai_correction：新网格模块 AI 修正子任务

- **任务编号**：mesh_05b_ai_correction
- **任务标题**：新网格模块 AI 修正：接口语义统一、基本算例跑通与旧类型依赖识别
- **版本**：v0.1
- **状态**：草案
- **创建时间**：2026-06-09
- **入库位置**：`kb/developments/new_mesh_module/mesh_05b_ai_correction/mesh_05b_ai_correction_task_brief.md`
- **启用条件**：当需要根据 2026-06-09 周例会行动项 A5，指导 AI Agent 集中修正新网格模块时，本文件必须启用
- **适用范围**：`fealpy/mesh` 新网格模块、`tests/mesh/unit/schema/` schema 单元测试、基础 mesher / 上层调用中对旧网格类型的显式依赖识别

本文档用于定义 `mesh_05b_ai_correction` 子任务的稳定本体表达，回答该子任务是什么、为什么成立、要修正哪些关键问题，以及 AI Agent 的完成判据是什么。

## 一、会议来源与行动项

本任务直接来自会议记录：

- 会议纪要：`D:/suanhai-repo/xihe/kb/meetings/2026/06/2026_06_09_group_weekly_meeting/meeting_notes.md`
- 关联议题：T7「网格模块接口边界：`grad_lambda`、形函数导数、`multi_index`、法向 / 切向、高维支持」
- 对应行动项：A5「网格模块集中推进：给出统一标准和策略，并让基本简单算例跑通」
- 截止时间：2026_06_16 前
- 关联决策：D4、D5

会议形成的最新口径如下：

1. 新网格架构目标是最终完全替换旧网格，不长期维护两套并行系统。
2. 上层应拿到兼容的 `mesh` 对象和接口；旧类型依赖可通过兼容层或假类型适配过渡。
3. 近期目标不是高维泛化，而是优先把 1D / 2D / 3D 常用情形做稳。
4. `grad_lambda` 是单纯形中线性形函数对实际坐标导数的特化，不应被误当成所有单元上的通用形函数导数接口。
5. 形函数及其导数要区分「参考单元导数」和「实际单元导数」。四边形、六面体等张量单元也应有形函数与导数，通常由一维基函数张量积构造。
6. `multi_index` 不只属于单纯形；张量单元也有局部编号 / 多重指标结构，并应允许不同方向阶数不同，默认可相同。
7. 法向 / 切向接口当前不必过度泛化；实体切向可先按默认标准方向返回，具体场景需要时再扩展。

## 二、Task 定位

`mesh_05b_ai_correction` 是 `mesh_05_algorithm_migration` 之后、系统性验收之前的 AI 修正子任务。它不是重新设计新网格模块，也不是完整替代 `mesh_03_optimize_architecture` 或 `mesh_05_algorithm_migration`，而是在最新会议决策下，对当前代码与资产中已经暴露的接口语义偏差、返回形状不一致、基本算例风险和旧类型依赖进行集中修正。

本任务的核心定位是：

- 把会议确定的接口标准落到 AI Agent 可执行的短周期任务中。
- 用最小范围修正支撑「基本简单算例跑通、常用接口可用」。
- 让 AI Agent 在改代码前先写回归测试，避免只凭直觉改动几何接口。
- 输出旧类型依赖清单，为后续统一替换和兼容层设计提供依据。

## 三、当前状态摘要

根据当前资产与代码初步检查，新网格模块已有以下基础：

1. 任务资产已存在：
   - 总任务资产：`kb/developments/new_mesh_module/`
   - 架构优化资产：`mesh_03_optimize_architecture/`
   - 算法迁移资产：`mesh_05_algorithm_migration/`
2. schema 代码已覆盖常见形状：
   - `fealpy/mesh/schema/node.py`
   - `fealpy/mesh/schema/edge.py`
   - `fealpy/mesh/schema/triangle.py`
   - `fealpy/mesh/schema/quadrilateral.py`
   - `fealpy/mesh/schema/tetrahedron.py`
   - `fealpy/mesh/schema/hexahedron.py`
   - `fealpy/mesh/schema/prism.py`
   - `fealpy/mesh/schema/pyramid.py`
3. 多数 schema 已有 `ccw`、`multi_index`、`barycenter`、`bc_to_point`、`measure`、`normal`、`tangent` 等方法。
4. 已有 schema 测试目录：`tests/mesh/unit/schema/`，包含 `test_triangle_schema.py`、`test_quadrilateral_schema.py` 等。
5. 已知口径冲突或需复核点：
   - `TriangleSchema.multi_index()` 当前接受 `int | tuple[int, ...]`，而 `mesh_05_algorithm_migration_team_handoff.md` 曾写明 `p` 参数应为 tuple；需要在本任务中明确是否保留兼容输入，以及文档与测试如何统一。
   - `QuadrilateralSchema.normal()` 当前返回 `(NC,)` 或 `(NC, GD)` 风格结果，而既有 handoff 曾要求 `normal()` / `tangent()` 统一为 `[实体数, 方向数, 几何维数]`。会议又提出法向 / 切向暂不宜过度泛化，因此本任务需要给出最小可执行策略：常用场景优先、返回形状一致、特殊含义记录清楚。
   - `PrismSchema` 中出现两个 `tangent` 方法定义，需检查是否为误覆盖。
   - `PyramidSchema` 包含形函数 / 几何映射相关方法，但不同 schema 之间的形函数导数命名尚未形成统一约定。
   - Line Walk 验证资产曾记录二维三角形几何量、局部边顺序与反向邻接等问题；其中几何接口问题已部分修正，但仍需通过单测确认。

## 四、核心问题与主目标

### 4.1 核心问题

当前风险不是「没有代码」，而是「接口语义和 AI 修正口径不统一」：

- `grad_lambda`、形函数导数、参考坐标导数、实际空间导数的边界容易被混淆。
- 单纯形与张量单元的 `multi_index` 规则容易被写成单一模式。
- 法向 / 切向返回形状容易在不同 schema 中不一致，导致上层调用无法稳定依赖。
- AI Agent 可能为了让测试通过而扩大改动范围，引入新 API 或破坏既有接口。
- 新旧网格替换路径中，显式旧类型依赖尚未形成清单。

### 4.2 主目标

本任务的主目标是：

1. 写清并落地新网格模块近期接口修正策略，优先服务 1D / 2D / 3D 常用情形。
2. 修正或标注 `grad_lambda`、形函数导数、`multi_index`、`normal()`、`tangent()` 的关键不一致点。
3. 让三类基本简单算例或等价测试跑通：
   - 1D edge 基本几何与插值点 / 多重指标；
   - 2D triangle / quadrilateral 基本几何、`bc_to_point`、`grad_lambda` / 几何导数；
   - 3D tetrahedron / hexahedron 至少一类基本几何与法向 / 切向形状验证。
4. 识别上层代码中显式依赖旧类型的导入与构造，例如 `from fealpy.mesh import TriangleMesh`、`TriangleMesh(...)`、`from fealpy.mesh_old ...`。
5. 输出可执行的 AI 修正记录与剩余风险清单，供后续 `mesh_07_integration_validation` 使用。

## 五、非目标与边界

本任务不做以下事项：

- 不重写整个新网格架构。
- 不把旧网格完整迁移完毕。
- 不为四维及以上几何维数设计完整泛化方案。
- 不新增大型抽象层或外部依赖。
- 不把所有上层模块一次性迁移到新网格。
- 不把形函数高阶有限元体系全部补齐；本任务只明确接口边界，并修正常用低阶 / 几何映射所需的最小能力。
- 不为了兼容旧接口而长期维持两套并行系统；兼容层或假类型只作为过渡策略记录。

## 六、AI Agent 工作原则

AI Agent 执行本任务时必须遵守以下原则：

1. 先读文档再改代码：至少阅读本 brief、`mesh_05_algorithm_migration_team_handoff.md`、`mesh_module_interface.md`、相关 schema 文件和对应测试文件。
2. 先写失败测试再修正：每个接口修正都应先有最小回归测试或现有失败测试作为依据。
3. 小步修正：每次只修正一个接口族或一个 schema，不做横跨架构的大改。
4. 兼容但不纵容旧接口：如保留 `int` 输入等兼容行为，必须在测试和文档中注明；不要让兼容行为掩盖标准接口。
5. 返回形状优先稳定：`normal()` / `tangent()` 面向上层使用时应尽量统一为 `[实体数, 方向数, 几何维数]`；若某 schema 暂不能满足，必须在风险记录中列出。
6. 常用维数优先：只要求 1D / 2D / 3D 常用情形；遇到 `GD > 3` 时可明确抛出 `NotImplementedError` 或 `ValueError`，不要伪造结果。
7. 旧类型依赖只识别不大改：本任务只要求形成清单与初步适配建议，除非为了基本算例必须做极小兼容修复。

## 七、建议执行子任务

### S1：建立接口修正基线

- 阅读：
  - `kb/developments/new_mesh_module/mesh_05_algorithm_migration/mesh_05_algorithm_migration_team_handoff.md`
  - `kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_module_interface.md`
  - `fealpy/mesh/schema/*.py`
  - `tests/mesh/unit/schema/*.py`
- 产出：当前 schema 方法与测试覆盖的简表。
- 验收：能列出每个 schema 是否具备 `multi_index`、`grad_lambda`、形函数相关方法、`normal`、`tangent`、`ccw`。

### S2：统一 `multi_index` 策略与测试

- 对单纯形：确认 `edge`、`triangle`、`tetrahedron` 的多重指标语义。
- 对张量单元：确认 `quadrilateral`、`hexahedron` 支持一个阶数默认各向相同，也支持各方向不同阶数。
- 对混合 / 特殊单元：`prism`、`pyramid` 若已实现则补测试；若未完整实现则记录限制。
- 验收：`tests/mesh/unit/schema/` 中相关测试能覆盖 tuple 输入、各向同性输入、各向异性输入和非法输入。

### S3：厘清 `grad_lambda` 与形函数导数边界

- 明确：`grad_lambda` 只表示单纯形 / 线性重心坐标的实际空间梯度，不能强行推广成所有形状的高阶形函数导数。
- 对 triangle / tetrahedron / edge：保留并测试 `grad_lambda`。
- 对 quadrilateral / hexahedron：若保留 `grad_lambda`，必须说明它只是当前几何映射下的线性 / 双线性节点形函数梯度近似或特定参考点结果；更推荐后续引入 `shape_function`、`grad_shape_function`、`transform_grad` 等明确命名。
- 对 prism / pyramid：检查已有 `shape_function`、`grad_shape_function`、`geometry_shape_function`、`geometry_grad_shape_function` 的命名与行为，先补最小文档和测试，不做全量高阶实现。
- 验收：测试或文档中能区分参考单元导数与实际空间导数。

### S4：修正 `normal()` / `tangent()` 返回形状不一致

- 目标返回形状：`[实体数, 方向数, 几何维数]`。
- 常用规则：
  - top_dim = T，geo_dim = G。
  - 切向数量优先为 T。
  - 法向数量优先为 `max(G - T, 0)`。
  - 当 `G == T` 时，实体本身无嵌入法向，允许返回形状 `(N, 0, G)` 的空方向张量。
- 当前优先检查：`EdgeSchema`、`TriangleSchema`、`QuadrilateralSchema`、`TetrahedronSchema`、`HexahedronSchema`。
- 已知重点：`QuadrilateralSchema.normal()` 当前测试期望可能是标量或向量，需改为统一三维形状或在文档中明确特殊返回；推荐优先修正到统一三维形状并更新测试。
- 验收：所有修改过的 schema 测试明确断言 shape，而不只断言数值。

### S5：跑通基本简单算例 / 等价单测

至少运行以下命令之一组，并保存结果：

```bash
python -m pytest tests/mesh/unit/schema/test_edge_schema.py -q
python -m pytest tests/mesh/unit/schema/test_triangle_schema.py -q
python -m pytest tests/mesh/unit/schema/test_quadrilateral_schema.py -q
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
python -m pytest tests/mesh/unit/schema/test_hexahedron_schema.py -q
```

若环境中 `pytest` 不可用，需记录阻塞原因，并使用可执行 Python 脚本做最小导入与接口调用验证。

### S6：识别旧类型依赖

搜索范围建议：

```bash
python - <<'PY'
from pathlib import Path
patterns = [
    'from fealpy.mesh import',
    'from fealpy.mesh_old',
    'TriangleMesh', 'QuadrangleMesh', 'TetrahedronMesh', 'HexahedronMesh',
    'EdgeMesh', 'IntervalMesh', 'PrismMesh', 'PyramidMesh'
]
for root in ['fealpy', 'app', 'example', 'test', 'tests']:
    for p in Path(root).rglob('*.py'):
        text = p.read_text(encoding='utf-8', errors='ignore')
        hits = [pat for pat in patterns if pat in text]
        if hits:
            print(p, '::', ', '.join(hits))
PY
```

产出建议文件：`mesh_05b_ai_correction_old_type_dependency_report.md`。

## 八、目标资产

本任务建议形成以下资产：

1. 必需资产：
   - `mesh_05b_ai_correction_task_brief.md`：本文件。
2. 建议资产：
   - `mesh_05b_ai_correction_agent_prompt.md`：给 AI Agent 的执行提示。
   - `mesh_05b_ai_correction_validation_checklist.md`：验收清单。
   - `mesh_05b_ai_correction_old_type_dependency_report.md`：旧类型依赖扫描结果。
   - `mesh_05b_ai_correction_execution_record.md`：执行记录、测试命令、失败与修复摘要。

## 九、对象级完成判据

当且仅当以下条件同时满足，本任务视为完成：

1. 已形成统一接口修正策略，覆盖 `grad_lambda`、形函数导数、`multi_index`、`normal()`、`tangent()`、1D / 2D / 3D 优先级和旧类型过渡策略。
2. 常用 schema 的 `multi_index` 行为有测试覆盖，且张量单元支持不同方向阶数或明确记录未支持原因。
3. `normal()` / `tangent()` 在已修正 schema 中返回形状一致，并有 shape 断言。
4. 至少 edge、triangle、quadrilateral、tetrahedron / hexahedron 中三类基本测试或算例跑通。
5. 旧类型显式依赖已扫描并形成清单，标注是否阻塞基本算例。
6. 所有代码修正均有对应测试或最小验证脚本。
7. 未引入四维以上泛化、大规模架构重写或与 A5 无关的功能扩展。

## 十、待确认点

1. `TriangleSchema.multi_index()` 是否继续允许 `int` 作为兼容输入，还是严格收敛为 `tuple[int, ...]`。
2. `QuadrilateralSchema.grad_lambda()` 是否应保留该名称，还是拆分为更清楚的参考形函数导数 / 实际空间导数接口。
3. 2D 面实体 `normal()` 返回空方向张量是否作为长期标准，还是仅作为当前过渡标准。
4. 兼容层或假类型适配应放在 `fealpy/mesh/__init__.py`、独立 compatibility 模块，还是上层调用处逐步迁移。

## 附录 A：版本演进记录

- **v0.1**：
  - 变更人：AI Agent
  - 变更时间：2026-06-09
  - 变更摘要：
    - 根据 2026-06-09 周例会 T7 / A5 建立 `mesh_05b_ai_correction` 子任务
    - 明确 AI 修正范围、接口策略、执行步骤、目标资产和完成判据
