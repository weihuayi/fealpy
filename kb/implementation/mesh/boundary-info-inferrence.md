# Boundary 信息推断实现框架

本文给出基于当前仓库拓扑构造结果的边界实体判定方法，目标是统一支持：

- 返回边界实体索引（index list）
- 返回边界实体布尔遮罩（bool mask）

并特别区分两类情况：

- 次高维实体（$D-1$ 维）：可直接通过“上层邻接计数”判定
- 更低维实体（$d < D-1$）：通常应通过边界闭包传播判定

## 一、与现有构造链路的对应关系

当前实现中，拓扑由以下链路构造：

- `TopologyBuilder.construct`
- `TopologyBuilder.construct_lower_dims`
- `_unique_unordered_rows_across`

核心结果是：

1. 每一类实体都在 `MeshStorage.blocks` 中有唯一编号（按行索引）。
2. 每对相邻层级实体之间写入 `MeshStorage.relations[(src_name, tgt_name)]`。
3. relation 的 `tgt_indices` 记录了源实体到目标实体的局部邻接索引表。

因此，边界判定不需要重新几何计算，只需要做 relation 上的计数和传播。

## 二、记号与输入

- 设网格拓扑维数为 $D$。
- 设某类实体类型为 $E$，其拓扑维数为 $\dim(E)$。
- 设 relation $R_{A\to B}$ 的目标索引张量为 `tgt_indices`。

约定输出：

- `mask_E`: 形状 `(N_E,)` 的 bool 张量
- `index_E`: `where(mask_E)` 的整数索引数组

## 三、次高维实体（$D-1$）边界判定

### 3.1 原理

对任意次高维实体（2D 网格中的 edge，3D 网格中的 tri/quad），其是否在边界可由“被多少个 $D$ 维实体引用”决定。

- 邻接计数为 1：边界实体
- 邻接计数为 2：流形内部实体
- 邻接计数 $>2$：非流形（可单独标记）
- 邻接计数为 0：孤立实体（通常表示数据异常或手工注入）

### 3.2 与本仓库数据结构的对齐

需要聚合所有满足以下条件的 relation：

- `src` 的 `top_dim == D`
- `tgt` 的 `top_dim == D-1`

原因：当前框架允许多个顶层 block（例如混合单元），同一种次高维实体可能同时被多个 `src` 类型引用，计数必须跨 relation 汇总。

### 3.3 伪代码

```python
def infer_boundary_codim1(storage, face_name):
	 # face_name: 一个具体次高维类型，如 "edge" / "tri" / "quad"
	 D = max(storage.get_block(name).schema.top_dim for name in storage.root_entity_names)
	 Nf = len(storage.get_block(face_name).indices)

	 count = bm.zeros((Nf,), dtype=bm.int32)

	 for (src, tgt), rel in storage.relations.items():
		  src_dim = storage.get_block(src).schema.top_dim
		  tgt_dim = storage.get_block(tgt).schema.top_dim
		  if src_dim == D and tgt_dim == D - 1 and tgt == face_name:
				idx = bm.reshape(rel.tgt_indices, (-1,))
				# 等价于 bincount + 累加
				count += bm.bincount(idx, minlength=Nf)

	 mask = (count == 1)
	 index = bm.where(mask)[0]
	 return index, mask, count
```

## 四、最高维实体（$D$）边界判定

### 4.1 判定定义

最高维实体（cell）的边界判定可定义为：

“若一个 $D$ 维实体至少引用一个边界次高维实体，则该 $D$ 维实体是边界实体。”

即在 relation $R_{D\to D-1}$ 上，若某行（某个 cell 的局部次高维实体索引）中存在边界次高维索引，则该 cell 为边界。

### 4.2 与当前数据结构对齐

1. 先得到某个次高维类型（如 `edge` / `tri` / `quad`）的 `mask_codim1`。
2. 读取对应 relation：`(src_name, codim1_name)`，其中 `src_name` 为最高维类型。
3. 对 relation 的每一行执行“是否命中边界次高维实体”的 `any` 判断。

对于混合单元，按每个最高维类型分别做上述判断，最终分别返回每类 cell 的边界索引和遮罩。

### 4.3 伪代码

```python
def infer_boundary_topdim_from_codim1(storage, src_name, codim1_name, mask_codim1):
	 rel = storage.relations[(src_name, codim1_name)]
	 cell2face = rel.tgt_indices  # shape: (Ncell, Nface_per_cell)

	 # cell 上任一局部次高维实体命中边界，即该 cell 为边界
	 cell_mask = bm.any(mask_codim1[cell2face], axis=1)
	 cell_index = bm.where(cell_mask)[0]
	 return cell_index, cell_mask
```

注：该定义与几何直觉一致。内部 cell 的所有次高维实体都应是内部实体；只要出现一个边界次高维实体，cell 必然贴近物理边界。

## 五、更低维实体（$d < D-1$）边界判定

### 5.1 不能直接用邻接计数作为判断标准

对边、点等更低维实体，直接统计其被多少个高维实体引用并不能稳定表达边界语义。

示例（3D）：

- 一个边界边通常会被多个体单元共享，计数可能远大于 1。
- 一个边界点也可能连接很多边界面和内部面。

因此，对 $d < D-1$ 的实体，推荐定义为：

“属于某个边界次高维实体的闭包”。

### 5.2 闭包传播法

步骤：

1. 先按第 3 节得到全部次高维边界 mask。
2. 沿 relation 逐层向低维传播：
	- 若父实体在边界，则其所有子实体都标记为边界。
3. 可按需传播到指定维数，或一直传播到 node。

### 5.3 伪代码

```python
def propagate_boundary_downward(storage, parent_name, parent_mask, child_name):
	 rel = storage.relations[(parent_name, child_name)]
	 Nc = len(storage.get_block(child_name).indices)

	 child_mask = bm.zeros((Nc,), dtype=bm.bool)
	 boundary_parent = bm.where(parent_mask)[0]
	 child_idx = bm.reshape(rel.tgt_indices[boundary_parent], (-1,))
	 child_mask[child_idx] = True
	 child_index = bm.where(child_mask)[0]
	 return child_index, child_mask
```

在全维度场景中，可按拓扑维数从 $D-1$ 到 0 做 BFS/分层传播。

## 六、统一接口建议

可将推断过程封装成两个层次：

1. `infer_boundary_codim1(storage) -> dict[name, BoundaryInfo]`
2. `infer_boundary_topdim(storage) -> dict[name, BoundaryInfo]`
3. `infer_boundary_entities(storage, target_dim | target_name) -> BoundaryInfo`

其中 `BoundaryInfo` 可包含：

- `index`: 边界索引
- `mask`: 边界布尔遮罩
- `count`（可选）: 上层邻接计数，仅对 $D-1$ 维有明确物理意义

## 七、复杂度与实现要点

### 7.1 复杂度

- 次高维计数：
  设所有 $D\to D-1$ relation 的条目总数为 $M$，复杂度约 $O(M)$（计数）
- 最高维判定：
	对每个 $D\to D-1$ relation 做逐行 `any`，总复杂度约 $O(M)$
- 闭包传播：
  设被访问的 relation 条目总数为 $P$，复杂度约 $O(P)$

总复杂度线性于关系表规模，适合大网格批处理。

### 7.2 注意事项

**混合网格**

必须跨所有顶层 `src` 类型累计同一 `tgt` 类型的计数。

**非流形处理**

建议保留 `count > 2` 的标记，便于网格质量诊断。

**方向无关性**

当前 builder 在构造下层实体时使用无序去重，边界判定天然不依赖局部节点方向。

**结果缓存**

若 relation 不变，建议把 `mask/index/count` 缓存到 block attributes ，避免重复计算。

## 八、小结

在本仓库现有拓扑设计下，边界推断可以完全建立在 relation 图上：

- 对次高维实体，用“高维邻接计数”直接判定。
- 对最高维实体，用“是否引用边界次高维实体”判定。
- 对更低维实体，用“从边界次高维实体向下取闭包”判定。

该方案同时支持输出边界索引与 bool mask，并兼容单一单元和混合单元场景。
