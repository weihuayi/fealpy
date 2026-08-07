# 实体构造链路实现说明

本文总结如下调用链的实现逻辑与设计意图：

`construct -> construct_lower_dims -> _unique_unordered_rows_across -> lexsort`

对应代码位于：
- `fealpy/mesh/topology/builder.py`

## 一、总览

该链路用于从高维单元（如体单元）逐层构造低维实体（如面、边），并建立相邻层级之间的关系映射。

核心目标：
- 合并多个来源 block 的候选低维实体；
- 对“节点顺序无关”的实体做去重；
- 返回每个来源 block 到去重后实体的逆映射；
- 在 `MeshStorage` 中更新实体块和关系。

## 二、`TopologyBuilder.construct`

`construct(storage, src_name=None)` 是整体驱动入口。

行为：
1. 决定起始 block 集合：
   - `src_name is None` 时，从 `storage.root_entity_names` 全部开始；
   - 否则仅从指定 block 开始。
2. 进入 while 循环，每轮调用 `construct_lower_dims(...)` 构造下一层实体。
3. 对每个返回的 `face_type_name`：
   - 若不存在则新增 `EntityBlock`；
   - 若已存在则覆盖其 `indices`。
4. 为每个源 block 写入关系：
   - key 为 `(block.schema_name, face_type_name)`；
   - value 为 `Relation(..., tgt_indices=cell2face)`。
5. 本轮得到的低维实体块作为下一轮 `current_blocks`，直到无法继续构造。

结果：
- 完成从高维到低维的分层构造；
- 每层都保留从上层到下层的索引映射关系。

## 三、`TopologyBuilder.construct_lower_dims`

`construct_lower_dims(cells, local_face_dicts)` 负责“一轮”低维实体构造，返回 `Iterator[ConstructResult]`。

### 3.1 输入语义

- `cells`: 每个元素是形状 `(NC, NVF)` 的单元节点索引张量。
- `local_face_dicts`: 与 `cells` 一一对应；每个字典提供：
  - key: 面类型名（例如 triangle / quadrilateral）；
  - value: 局部面定义（节点局部下标列表）。

### 3.2 主要步骤

1. 按 `face_kind` 聚合候选面：
   - 通过 `get_total_face(cell, local_face)` 生成 `(总面数, NFC)` 的面节点张量；
   - 存入 `face_table[face_kind][0]`；
   - 同时记录每个来源块的 `len(local_face)` 到 `face_table[face_kind][1]`。
2. 对每个 `face_kind` 调用 `_unique_unordered_rows_across(*total_face_list)`：
   - 得到去重后 `face`；
   - 得到每个来源数组对应的逆映射 `js`（来源数组中每个元素在去重数组中的索引）。
3. 把每个 `j` reshape 成 `(-1, NFC)`，得到 `cell2faces`。
4. 抛出结果 `yield ConstructResult(face_kind, face, cell2faces)`。

## 四、`_unique_unordered_rows_across` 的去重机制

该函数是本链路正确性的关键。

函数签名：
- `_unique_unordered_rows_across(*arrays: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]`

约束：
- 至少一个输入数组；
- 每个输入必须是二维张量。

### 4.1 为什么需要“无序去重”

对于面实体，节点序列常因单元局部方向不同而出现排列差异。
例如：
- `[1, 5, 9]` 与 `[9, 1, 5]`

几何上是同一个面，但逐行严格比较会误判成不同实体。

### 4.2 算法步骤

**第一步**  纵向拼接
   - `total = concat(arrays, axis=0)`。把每一种高维实体产生的低维实体拼在一起。

**第二步**  行内规范化（仅用于排序键）
   - `canonical_total = sort(total, axis=1)`。

**第三步**  基于规范化键做字典序排序
   - `indices = lexsort(reversed(canonical_total.T), axis=0)`；
   - 这是 sorted -> original 的映射。

**第四步**  在排序后的规范化数组上检测组边界
   - `diff_flag[0] = True`；
   - `diff_flag` 的其余位置比较当前行与前一行是否不同 —— `True` 的位置，就是唯一元素即将产生的位置。

**第五步**  生成唯一代表
   - `unique = total[indices[diff_flag]]`；
   - 其中 `indices[diff_flag]` 是 unique -> original 的映射 —— 每个 `unique` 元素在 `total` 中第一次出现的位置；
   - 注意这里使用原始 `total`，因此 `unique` 内每一行保持原始节点顺序。

**第六步**  构造逆映射
   - `sorted_to_unique = cumsum(diff_flag) - 1`（排序序列到唯一序列）；
   - 先建立 `original_to_sorted`（原始序到排序序）；
   - 再得到 `total_to_unique = sorted_to_unique[original_to_sorted]`；
   - 这一步是两个映射的复合。

**第七步**  按输入数组切分逆映射，返回 `arr_to_unique`。

### 4.3 输出语义

返回 `(unique, arr_to_unique)`：
- `unique`: 去重后的实体节点数组（每行保留原始顺序表示）；
- `arr_to_unique[i]`: 第 `i` 个输入数组到 `unique` 的索引映射。

## 五、`lexsort` 在链路中的角色

`lexsort` 提供稳定、可控的多关键字间接排序能力。

在此处：
- 关键字是 `canonical_total` 的各列（逆序传入以符合字典序）；
- 排序后，相同规范化行相邻，便于线性扫描提取唯一组；
- 进一步支持高效生成“原始索引 -> 唯一索引”的逆映射。

## 六、正确性要点

1. 无序等价正确：
   - 同一面的任意节点排列在规范化后键相同，必归并到同一唯一组。
2. 表示稳定：
   - 唯一组代表来自首次出现的原始行，保留原顺序。
3. 映射完整：
   - 对每个输入数组都返回可重建其实体引用的逆映射。

## 七、复杂度概览

设总候选行数为 `M`，每行列数为 `K`。

- 行内排序：约 `O(M * K log K)`
- 行间字典序排序：约 `O(M log M)`（比较代价与 `K` 相关）
- 线性扫描与映射恢复：`O(M)`

总体瓶颈通常在排序阶段。

## 八、维护建议

1. 若将来支持三维以上输入，应明确“按哪一维作为实体行”的规范并补齐测试。
2. 若需要控制唯一代表的方向一致性（如统一逆时针），应在去重后增加方向规范阶段，而不是放在当前函数中。
3. 建议补充回归用例：
   - 同一面不同排列应映射到同一索引；
   - `unique` 的每行保持原始顺序；
   - 多输入数组切分映射边界正确。
