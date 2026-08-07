# N-03 | Schema 接口契约冻结

- 任务：`mesh_schema_interface_test_refactor`
- 节点：N-03 契约冻结
- 版本：v0.1
- 状态：冻结可执行契约；保留明确开放项
- 输入：`assets/N-01_schema_contract_inventory.md`、`kb/design/mesh/mesh_entity_functions.md`、`fealpy/mesh/schema/entity_schema.py`
- 输出用途：直接约束 N-04 `schema_cases` 数据模型、能力声明和独立参考值

## 1. 冻结范围与优先级

本文件冻结“测试数据和统一接口测试”需要的可执行契约，不冻结生产实现，也不把未决事项伪装成已实现能力。

优先级：

1. `kb/design/mesh/mesh_module_contract.md`：形状、顶点、`OFace`、`SFace` 和局部顺序。
2. `kb/design/mesh/mesh_entity_functions.md`：标准接口集合及语义。
3. `fealpy/mesh/schema/entity_schema.py`：实际 Python 签名、参数名和返回约定补充。
4. 当前实现、现有测试：只作为观察证据和偏差来源。

## 2. Case 数据模型冻结

N-04 每个 case 必须提供或可推导以下字段：

| 字段                        | 类型/约束                                      | 冻结规则                                                         |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| `case_id`                 | 唯一字符串                                      | 包含几何族、形状和场景；失败报告必须显示它                                        |
| `schema_name`             | 枚举                                         | `point`、`segment`、`tri`、`quad`、`tet`、`prism`、`pyramid`、`hex` |
| `positions`               | 固定 Tensor/可转换数组                            | 形状 `(NN, GD)`；不得随机生成                                         |
| `indices`                 | 整数 Tensor                                  | 形状 `(NC, NV)`；列数必须匹配 schema 顶点数                              |
| `top_dim`                 | 整数                                         | 与 schema 注册值相同                                               |
| `geo_dim`                 | 整数                                         | 等于 `positions.shape[1]`，允许 `geo_dim > top_dim`               |
| `geometry_kind`           | `regular`/`distorted`                      | 至少各有一类适用 case                                                |
| `entity_count`            | 整数                                         | 等于 `indices.shape[0]`                                        |
| `index_cases`             | `None`、整数/整数数组、slice 等                     | 为支持 index 的接口提供全量和子集证据                                       |
| `supported_interfaces`    | 不重复接口名集合                                   | 统一入口只依此筛选；不在测试函数中复制 shape 判断                                 |
| `backend_status`          | backend -> `supported`/`skip`/`unverified` | 明确能力，不用静默跳过                                                  |
| `expected_local_entities` | `OFace`/`SFace` 映射                         | 来自 N-01 规范，不从 schema class 复制                                |
| `bcs`、`polynomial_orders` | 固定测试输入                                     | 与参考实体和接口签名匹配                                                 |
| `references`              | 独立计算/期望值                                   | 非仿射积分不调用生产 `integral` 生成 oracle                              |

建议 `SchemaCase` 提供 `build_context(backend)`，统一构造 `MeshBlock`、`EntitySector` 和 `EntityContext`；case 定义与测试断言分离。

## 3. 接口分组与最小断言契约

### 3.1 拓扑接口

| 接口             | 必须断言                                                      | 适用性           |
| -------------- | --------------------------------------------------------- | ------------- |
| `boundary`     | `index`、`mask` 类型/长度/一致性；边界索引可回到实体集合                      | 有实体边界时        |
| `local_entity` | `o/s` 返回值等于规范局部实体；非法 indexing 明确报错                        | 所有非 point 根实体 |
| `relation`     | 目标 relation 的 source/target 数量和索引范围；方向与 `local_entity` 对应 | 目标实体存在时       |
| `size`         | 等于 sector entity count；index 不改变 sector size 语义           | 所有 case       |

### 3.2 多重指标与朝向接口

| 接口 | 必须断言 | 适用性 |
|---|---|---|
| `multi_index` | 输出为二维；列语义正确；行数等于 `num_multi_index`；阶数守恒 | 支持给定 order 的 case |
| `num_multi_index` | 等于 `multi_index` 行数 | 与 `multi_index` 同步 |
| `global_permutations` | 置换值在合法顶点范围，局部实体节点经置换后与全局关系一致 | 至少多实体共享低维实体 case |
| `vo_to_do` | 每个声明的 orientation 都有映射；映射无重复且范围合法 | case 声明支持时 |

### 3.3 几何、映射、微分与积分接口

| 接口 | 输出形状/数值断言 | 备注 |
|---|---|---|
| `barycenter` | `(NC_selected, GD)`；规则 case 与独立顶点平均/几何定义一致 | 检查 `None` 与 subset |
| `barycentric` | 装饰后的函数接收 bcs 并等于直接物理坐标函数 | 不重复实现映射 oracle |
| `bc_to_point` | 前导实体维度、积分点维度和 GD 正确；规则映射值可独立计算 | 张量积 shape 必须记录 |
| `geo_dimension` | 等于 `positions.shape[1]` | 所有 case |
| `grad_shape_function_barycentric` | 文档形状 `(NQ, num_shape, num_bc)`；适用时分割统一性/导数和为零 | 参考坐标无关物理变形 |
| `grad_shape_function_reference` | 文档形状 `(NQ, num_shape, ref_dim)`；与形函数节点/阶数对应 | 不与 production helper 互作 oracle |
| `grad_shape_function_cartesian` | 输出含 GD 维；通过独立链式规则检查简单规则 case | 需要 Jacobian |
| `jacobi_matrix` | `(NC_selected, NQ, GD, ref_dim)`；规则 affine case 常量 | 非仿射 case 允许随 q 变化 |
| `measure` | 每个 selected entity 一个有限非负值；规则几何与解析测度一致 | quad/hex 非仿射用独立参考 |
| `normal` | 形状/嵌入维数正确；方向/有向测度按 case 规范检查 | 2D/3D 语义须显式声明 |
| `tangent` | 切向数量和 GD 维正确；方向/长度按 case 记录 | 不把空 tangent 当普遍契约 |
| `integral` | 常数/低阶函数与独立解析或数值参考一致；输出实体前导维度 | 非仿射必须独立 Jacobian 测度 |
| `quadrature_formula` | bcs 是 tuple；tuple 长度与参考维数一致；weights 是一维且点数匹配 | 检查 q/qtype 能力 |
| `shape_function` | 输出点数与 shape 数；线性节点插值和分割统一性（适用时） | 高阶 case 只要求声明的性质 |

## 4. 能力矩阵（冻结版）

符号：`R`=必须纳入统一入口；`C`=条件纳入，由 case 能力声明；`S`=保留专项/暂不纳入；`U`=契约未决，禁止猜测；`-`=不适用。

| 接口族 | point | segment | tri | quad | tet | hex | prism | pyramid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 拓扑：`size`/`local_entity` | R | R | R | R | R | R | R | R |
| 拓扑：`boundary`/`relation` | C | R | R | R | R | R | C | C |
| `multi_index`/`num_multi_index` | C | R | R | R | R | R | C | C |
| `global_permutations`/`vo_to_do` | - | C | C | C | C | C | C | C |
| `barycenter`/`geo_dimension` | C | R | R | R | R | R | C | C |
| `bc_to_point`/`barycentric` | - | R | R | R | R | R | C | C |
| `shape_function` | C | R | R | R | R | R | C | C |
| gradient interfaces | - | C | R | R | R | R | C | C |
| `jacobi_matrix` | - | C | R | R | R | R | C | C |
| `measure`/`integral` | C | R | R | R | R | R | C | C |
| `normal`/`tangent` | - | C | C | R | C | C | C | C |
| `quadrature_formula` | C | R | R | R | R | R | C | C |

能力矩阵不是要求形成形状与接口的笛卡尔积。`C` 必须在具体 case 中给出 `capability_reason`；当前实现缺少方法时不得通过 `getattr` 失败掩盖。

## 5. 统一失败分类与回流

| 分类 | 判断 | 回流 |
|---|---|---|
| `CONTRACT_AMBIGUOUS` | 规范/接口文档无法决定期望 | 回 N-03，暂停断言冻结 |
| `IMPLEMENTATION_DIVERGENCE` | 规范明确、代码行为不符 | Schema 生产实现任务；保留最小失败 case |
| `CASE_DEFECT` | positions/indices/reference 错误 | N-04 数据模块 |
| `TEST_DEFECT` | 测试断言、筛选或 shape 检查错误 | N-05/统一入口维护者 |
| `UNSUPPORTED` | case 声明不适用且有依据 | 覆盖矩阵记录，不算绿色覆盖 |
| `BACKEND_GAP` | backend 未验证或行为不一致 | backend/测试系统承接 |

## 6. 开放项（显式不冻结）

1. 三维形状完整 `orientation` 集合未在设计合同中给出；N-04 只使用已冻结的 `OFace/SFace`，不生成未经依据的全排列。
2. prism、pyramid 各标准接口的最终适用集合仍需按实现能力和规范裁决；用 `C/U` 表示，不得静默跳过。
3. `normal`、`tangent` 在不同拓扑维数/嵌入维数的精确输出维度和方向约定需在具体 case 中补充独立数学依据。
4. 支持 backend 的最终清单和最小矩阵待项目测试环境确认。
5. point 的部分几何/积分语义不应由形状名称推断，需以实际接口契约单独登记。

## 7. N-04 开始条件

- [ ] 使用 `assets/N-01_schema_contract_inventory.md` 的顶点和 OFace/SFace 数据。
- [ ] 使用本文件的 `SchemaCase` 字段和 `supported_interfaces` 约定。
- [ ] 每一类 `C/U` 能力均有显式理由或开放项记录。
- [ ] 规则、非规则、嵌入、多实体、index 子集至少各有一个规划入口。
- [ ] quad/hex 的顺序和非仿射独立参考要求在 case metadata 中可见。
- [ ] 不修改生产 Schema；发现偏差时按分类回流并保留证据。
