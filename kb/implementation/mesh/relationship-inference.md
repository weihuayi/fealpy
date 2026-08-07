# 拓扑关系推断（新版设想）

## 1. 目标与动机

本文提出 `TopologyInferer` 的新推断算法，用于替代当前“逐行去重”的实现路径。主要目标：

- 避免 `for row in tgt_indices` 这类 Python 级大循环；
- 避免对每一行重复调用 `bm.unique_all(row)`；
- 在 homogeneous 关系下，仅使用张量索引与重排完成推断；
- 保持现有 API 行为（输入输出语义与异常语义不变）。

适用场景示例：已知

- `prism -> tri`
- `tri -> edge`
- `prism -> quad`
- `quad -> edge`

推断 `prism -> edge`。


## 2. 问题抽象

设：

- `src -> mid` 的目标宽度为 `M`（每个 src 实体关联 `M` 个 mid）；
- `mid -> dst` 的目标宽度为 `K`（每个 mid 实体关联 `K` 个 dst）；
- 组合后平铺宽度是 `W = M * K`。

传统复合得到 `(NC, W)` 后，行内通常出现重复 dst 索引，因此再做“每行 unique 保序”。瓶颈在于：

- 行循环是 Python 层；
- unique 调用次数与 `NC` 成正比。


## 3. 核心观察

在 homogeneous 拓扑中，**行内重复模式来自局部模板，而不是来自具体单元编号**。

换句话说：

- 对固定三元组 `(src_type, mid_type, dst_type)`；
- 组合后 `W` 个候选槽位中，哪些槽位是“首次出现”的位置是固定的；
- 该位置集合可预计算并缓存，运行时直接按列选取，无需逐行 `unique`。


## 4. 新算法：LPC（Local Pattern Composition）

### 4.1 总体流程

1. 预计算 `CompositionPlan(src, mid, dst)`；
2. 推断时执行一次张量复合：
	 - `raw = mid_to_dst.tgt_indices[src_to_mid.tgt_indices]`，形状 `(NC, M, K)`；
	 - `flat = reshape(raw, (NC, M*K))`；
3. 用 `plan.select_pos` 做列选择：
	 - `out = flat[:, select_pos]`，得到 `(NC, L)`；
4. 直接写入 `Relation(src, dst, out)`。

其中 `L` 是 `dst` 的真实局部个数（如棱柱到边是 9）。

### 4.2 预计算 `select_pos`

预计算基于“局部编号签名”而非全局实体索引：

1. 枚举组合槽位 `(i, j)`，其中 `i in [0, M)`, `j in [0, K)`；
2. 将槽位映射为 `src` 的局部节点集合签名（建议排序后 tuple）；
3. 对签名序列做“首次出现保序去重”；
4. 保留对应槽位的线性下标 `i*K + j`，形成 `select_pos`。

`select_pos` 对同一 `(src, mid, dst)` 固定，可放入缓存重复使用。


## 5. 数据结构设计

建议新增轻量结构：

```python
class CompositionPlan(NamedTuple):
		src_name: str
		mid_name: str
		dst_name: str
		select_pos: Tensor  # 1D int tensor, shape (L,)
		out_width: int      # L
```

缓存建议：

- 键：`(src_name, mid_name, dst_name)`；
- 值：`CompositionPlan`；
- 生命周期：`TopologyInferer` 类级缓存或 `MeshStorage` 级缓存均可。


## 6. 与现有 `TopologyInferer` 的集成

保持现有 `infer()` 调度逻辑，仅替换“组合+去重”步骤：

- 保留 `_infer_from()` 的层次推进策略；
- 将 `_compose_homogeneous()` 改为：
	- 先做张量复合与 reshape；
	- 再按 `CompositionPlan.select_pos` 列采样；
- 删除 `_deduplicate_rows_preserve_order()` 及其调用路径。

等价替换点：

- 旧：`compose -> merge -> row-wise unique`；
- 新：`compose -> fixed-column-select`。


## 7. 伪代码

```python
def compose_homogeneous_fast(storage, src_name, mid_name, dst_name, src_to_mid, mid_to_dst):
		plan = get_or_build_plan(storage, src_name, mid_name, dst_name)

		raw = mid_to_dst.tgt_indices[src_to_mid.tgt_indices]      # (NC, M, K)
		flat = bm.reshape(raw, (len(src_to_mid.tgt_indices), -1)) # (NC, M*K)
		out = flat[:, plan.select_pos]                             # (NC, L)
		return out


def get_or_build_plan(storage, src_name, mid_name, dst_name):
		key = (src_name, mid_name, dst_name)
		if key in PLAN_CACHE:
				return PLAN_CACHE[key]

		signatures = []
		# 由 schema 的 local 关系构造签名序列；每个签名表示一个候选槽位在 src 局部点集上的等价类
		for i in range(M):
				for j in range(K):
						signatures.append(local_signature(src_name, mid_name, dst_name, i, j))

		select_pos = first_occurrence_positions(signatures)  # 1D int list/tensor
		plan = CompositionPlan(src_name, mid_name, dst_name, bm.asarray(select_pos, dtype=bm.int32), len(select_pos))
		PLAN_CACHE[key] = plan
		return plan
```


## 8. 复杂度对比

记 `NC` 为源实体数量。

- 旧方案：`O(NC * W log W)`（逐行 unique，且包含 Python 行循环）；
- 新方案：`O(NC * W)`（纯张量 gather + reshape + 列选）；
- 预计算：`O(W)` 到 `O(W log W)`，仅首次一次性发生。

对大 `NC`，新方案可显著降低 Python 解释器开销与重复 unique 成本。


## 9. 正确性约束与前提

1. 仅适用于 homogeneous relation：`Relation.src_indices is None`；
2. `select_pos` 必须由局部拓扑模板推导，不能由某个实例单元“猜测”；
3. 同一 `(src, mid, dst)` 的 `out_width` 固定；
4. 若 plan 构造失败（缺局部拓扑信息），应回退到旧路径或抛出明确异常。


## 10. 以三棱柱为例（prism -> tri/quad -> edge）

- `M = 5`（棱柱 5 个面：2 三角 + 3 四边形）；
- 对三角面链路 `K = 3`，对四边形面链路 `K = 4`（先按照 schema 中 local_face 字典中的顺序拼接，再利用 plan 选取）；
- 平铺候选中存在重复边；
- 通过预计算 `select_pos` 直接取 9 个唯一边槽位，得到 `prism -> edge`。

注意：若中间 `mid` 存在多个子类型（如三角面、四边形面），应先按顺序拼接，再构建 plan，因为三角面和四边面之间存在共享边。


## 11. 工程落地建议

对于多边形、多面体网格，存在完全不同的数据结构（如 half-edge、组合映射等）和完全不同的拓扑构建方式。因此，本文描述的实现将作为 homogeneous 类型网格拓扑推断的唯一实现，**建议完全替换旧方法**。

适用源形状：线、三角形、四边形、四面体、三棱柱、四棱锥、六面体。

## 12. 测试建议

- 正确性：
	- 与旧算法结果逐项对比（同一输入，输出集合一致、顺序策略一致）；
	- 覆盖 `tet`、`prism`、多单元共享实体场景。

- 鲁棒性：
	- 缺失 plan 信息时的异常或 fallback 行为；
	- heterogeneous relation 触发拒绝逻辑。

- 性能：
	- 大规模 `NC` 下比较 wall time；
	- 关注 Python profile 中 row-loop/unique 热点是否消失。


## 13. 非目标

- 本阶段不处理 heterogeneous relation 的通用组合语义；
- 不改变 `TopologyInferer.infer(storage, src_name, dst_name)` 对外接口；
- 不引入与关系推断无关的数据结构重构。
