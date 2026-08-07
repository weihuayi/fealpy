# FEALPy 网格模块接口参考

## 一、实体 Schema 的类方法与类属性

### 实体拓扑方法

| 方法名            | 参数            | 返回              | 用途                     |
| -------------- | ------------- | --------------- | ---------------------- |
| `boundary`     | ctx           | BoundaryInfo    | 获取该种形状的实体的边界信息。        |
| `local_entity` | tgt_name      | list[list[int]] | 获取用于构造局部子实体的索引。        |
| `relation`     | ctx, tgt_name | Relation        | 获取该实体与另一种形状的实体之间的拓扑关系。 |
| `size`         | ctx           | int             | 获取该种形状的实体的数量。          |

### 多重指标方法

| 方法名                | 参数  | 返回     | 用途                   |
| ------------------ | --- | ------ | -------------------- |
| `multi_index`      |     | Tensor | 计算该种实体上的多重指标。        |
| `multi_index_sort` |     | Tensor | 对该种实体上的已知多重指标进行稳定排序。 |
| `num_multi_index`  |     | int    | 获取多重指标数量。            |

### 几何计算方法

| 方法名                  | 参数名              | 返回类型   | 用途             |
| -------------------- | ---------------- | ------ | -------------- |
| `barycenter`         | ctx, index       | Tensor | 计算该种形状实体的重心。   |
| `bc_to_point`        | ctx, bcs,  index | Tensor | 把重心坐标转化为笛卡尔坐标。 |
| `geo_dimension`      | ctx              | int    | 获取几何维数。        |
| `grad_lambda`        | ctx, index       | Tensor | 重心坐标对笛卡尔坐标的梯度。 |
| `quadrature_formula` | q, qtype         | 积分公式   | 在该种形状实体上的积分公式。 |
| `measure`            | ctx, index       | Tensor | 计算该种形状实体的测度。   |
| `normal`             | ctx, index       | Tensor | 计算该种形状实体的法向。   |
| `tangent`            | ctx, index       | Tensor | 计算该种形状实体的切向。   |

## 二、实体视图的实例方法

| 方法名                 | 参数类型               | 返回类型               | 用途                                             |
| ------------------- | ------------------ | ------------------ | ---------------------------------------------- |
| `barycenter`        | index              | Tensor             | 计算实体重心                                         |
| `barycentric`       | (Tensor) -> Tensor | (Tensor) -> Tensor | 把笛卡尔坐标函数（Cartesian）转化为该实体上的重心坐标函数（Barycentric） |
| `boundary`          | -                  | BoundaryInfo       | 获取边界信息                                         |
| `geo_dimension`     | -                  | int                | 获取实体的几何维数                                      |
| `grad_lambda`       | index              | Tensor             | 计算重心坐标对笛卡尔坐标的梯度                                |
| `(property)indices` | -                  | Tensor             | 获取实体对节点的索引                                     |
| `measure`           | index              | Tensor             | 计算实体测度                                         |
| `normal`            | index              | Tensor             | 计算实体法向                                         |
| `size`              | -                  | int                | 获取实体数量                                         |
| `tangent`           | index              | Tensor             | 计算实体切向                                         |
| `to`                | EntityView         | Relation           | 获取实体到其他实体的拓扑关系                                 |
| `top_dimension`     | -                  | int                | 获取拓扑维数                                         |

## 三、网格视图的实例方法

### 新接口

| 方法名             | 参数           | 返回             | 用途                                              |
| --------------- | ------------ | -------------- | ----------------------------------------------- |
| `entity_count`  | 实体维数         | int            | 获取具有指定拓扑维数的实体的总数。                               |
| `entities`      | 实体维数         | Tensor 迭代器     | 迭代具有指定维数的实体的索引/坐标，获取同于旧式接口中 `cell`、`node` 等的张量。 |
| `entity_views`  | 实体维数         | EntityView 迭代器 | 指定维数，获取所有实体视图。                                  |
| `geo_dimension` | -            | int            | 几何维数。                                           |
| `relation`      | 源实体维数、目标实体维数 | Relation 迭代器   | 遍历所有符合维数要求的源实体到目标实体的拓扑关系。                       |
| `sector`        | 实体形状名称       | EntityView     | 指定形状类型，获取实体视图。                                  |
| `top_dimension` | -            | int            | 拓扑维度。                                           |

### 过渡接口

指为了兼容经典接口而引入的工具接口。

| 方法名           | 参数  | 返回         | 用途                                         | 依赖             |
| ------------- | --- | ---------- | ------------------------------------------ | -------------- |
| `entity_view` |     | EntityView | 指定维数，获取**唯一的**实体视图。只适用于任意拓扑维度仅存在一种形状的经典网格。 | `entity_views` |

### 经典接口

| 方法名                  | 参数  | 返回  | 用途  | 依赖            |
| -------------------- | --- | --- | --- | ------------- |
| `entity`             |     |     |     | `entity_view` |
| `entity_barycenter`  |     |     |     |               |
| `number_of_cells`    |     |     |     |               |
| `number_of_faces`    |     |     |     |               |
| `number_of_edges`    |     |     |     |               |
| `number_of_nodes`    |     |     |     |               |
| `multi_index_matrix` |     |     |     |               |
| `quadrature_formula` |     |     |     |               |
