# FEALPy | Validation Checklist | mesh_05b_ai_correction

本文档用于验收 `mesh_05b_ai_correction` 子任务是否满足 2026-06-09 周例会 T7 / A5 的要求。

## 一、来源一致性

- [x] 已明确引用会议纪要 T7 / A5。
- [x] 已落实 D4：新网格最终替换旧网格，不长期维护两套并行系统。
- [x] 已落实 D5：近期不为四维及以上过度泛化，优先 1D / 2D / 3D。

## 二、接口语义

### `grad_lambda` 与形函数导数

- [ ] 文档或代码注释已说明 `grad_lambda` 是单纯形线性重心坐标实际空间梯度的特化。
- [ ] 没有把 `grad_lambda` 作为所有单元 / 所有阶数形函数导数的泛化接口强行使用。
- [x] 对参考单元导数与实际空间导数有明确区分。
- [x] 张量单元如保留 `grad_lambda`，已说明语义限制或后续替代接口。

### `multi_index`

- [ ] 单纯形单元的 `multi_index` 有测试覆盖。
- [ ] 张量单元的 `multi_index` 支持各向同性阶数输入。
- [ ] 张量单元的 `multi_index` 支持各方向不同阶数输入，或明确记录未支持原因。
- [ ] 非法输入有测试或明确异常行为。

### `normal()` / `tangent()`

- [ ] 修改过的 schema 中，`normal()` 返回值 shape 已测试。
- [ ] 修改过的 schema 中，`tangent()` 返回值 shape 已测试。
- [ ] 返回形状尽量统一为 `[实体数, 方向数, 几何维数]`。
- [ ] `G == T` 的空法向情况有明确测试或说明。
- [ ] 不为 `GD > 3` 伪造未验证结果。

## 三、基本算例 / 测试

至少满足以下三类：

- [x] 1D edge 基本几何、`bc_to_point` / `multi_index` 可用。
- [x] 2D triangle 或 quadrilateral 基本几何、`bc_to_point`、`grad_lambda` / 等价几何导数可用。
- [x] 3D tetrahedron 或 hexahedron 基本几何、`normal()` / `tangent()` shape 可用。

推荐命令已运行并记录真实结果：

```bash
python -m pytest tests/mesh/unit/schema/test_edge_schema.py -q
python -m pytest tests/mesh/unit/schema/test_triangle_schema.py -q
python -m pytest tests/mesh/unit/schema/test_quadrilateral_schema.py -q
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
python -m pytest tests/mesh/unit/schema/test_hexahedron_schema.py -q
```

## 四、旧类型依赖识别

- [x] 已扫描 `fealpy/`、`app/`、`example/`、`test/`、`tests/` 中的旧类型依赖。
- [x] 已形成 `mesh_05b_ai_correction_old_type_dependency_report.md`。
- [x] 报告区分阻塞基本算例的依赖与非阻塞遗留依赖。
- [x] 对兼容层 / 假类型适配仅提出过渡建议，没有将其作为长期并行路线。

## 五、变更纪律

- [x] 每个代码修正有对应测试或最小验证脚本。
- [x] 没有新增外部依赖。
- [x] 没有大规模重写架构。
- [x] 没有展开四维及以上泛化。
- [x] 没有把任务扩展到全量上层模块迁移。

## 六、最终交付

- [x] 代码修改清单完整。
- [x] 测试命令与真实输出已记录。
- [x] 剩余风险与待确认点已记录。
- [x] 满足 A5：常用接口可用；基本简单算例跑通；接口约定清楚；旧类型依赖已识别。
