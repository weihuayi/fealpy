# 计算视图接口表

> [!CAUTION] 
> 不实现算法：计算视图不承担算法实现，只做算法以外的入参解析、算法调用、资源调度、输出微调/修饰。

> [!WARNING]
> 解耦：计算视图不区分实体类型、网格类型；网格和实体也不区分计算视图。

## 一、实体视图接口

> [!WARNING]
> 实体接口必须具有“只对某种实体取得属性/执行计算，其它实体不受影响”的语义。正例：形函数；反例：均匀加密。

### 1.1 一般实体视图 EntityView

> [!CAUTION]
> 一般实体视图不为网格细分领域算法开放接口，这些算法应实现为外部函数，或者自定义计算视图。

| 函数                    | 功能                   |
| --------------------- | -------------------- |
| `barycentric`         | 装饰器，把笛卡尔坐标函数变为重心坐标函数 |
| `barycenter`          | 计算单元重心               |
| `bc_to_point`         | 重心坐标转换为笛卡尔坐标         |
| `boundary`            | 获得边界信息               |
| `del_attribute`       | 删除实体上的属性             |
| `error`               | 计算两个函数之间的误差          |
| `geo_dimension`       | 几何维数                 |
| `get_attribute`       | 获取实体上的属性             |
| `global_permutations` | 子实体局部朝向到全局朝向的顶点置换矩阵  |
| `grad_shape_function` | 计算形函数的梯度             |
| `indices`             | 属性，单元顶点的全局编号         |
| `integral`            | 计算函数积分               |
| `jacobi_matrix`       | 雅可比矩阵                |
| `measure`             | 计算单元测度               |
| `multi_index_matrix`  | 获取多重指标，0-轴广播形式/张量积形式 |
| `normal`              | 计算单元法向               |
| `num_multi_index`     | 计算多重指标数量             |
| `quadrature_formula`  | 获取积分公式               |
| `set_attribute`       | 设置实体上的属性             |
| `shape_function`      | 计算形函数值               |
| `size`                | 获取实体数量               |
| `tangent`             | 计算单元切向               |
| `to`                  | 获取到另一实体的拓扑关系         |
| `top_dimension`       | 获取拓扑维数               |

> [!NOTE]
> 代码文件 `fealpy/mesh/view/entity_view.py`

## 二、网格视图接口

> [!WARNING]
> 网格接口必须具有“面向整个网格/多种实体获取信息/执行计算”的语义。

### 2.1 一般网格视图 Mesh

> [!CAUTION]
> 一般网格视图不为网格细分领域算法开放接口，这些算法应实现为外部函数，或者自定义计算视图。

| 函数                | 功能               | 备注                             |
| ----------------- | ---------------- | ------------------------------ |
| `Entity`          | 按拓扑维数/形状名字获取实体视图 | 接收多种形式的参数                      |
| `Entities`        | 按拓扑维数获取多个实体视图    |                                |
| `geo_dimension`   | 获取几何维数           | 与位置张量的列数一致。                    |
| `top_dimension`   | 获取拓扑维数           | 没有根实体时返回 `-1`。                 |
| `is_simplex_mesh` | 判断是否为单纯形网格       | 仅检查 sector 名称集合，不做几何合法性验证。     |
| `is_tensor_mesh`  | 判断是否为张量网格        | 仅检查 sector 名称集合，不做几何合法性验证。     |
| `is_elemental`    | 判断是否只有一个根实体类型    | 可选传入 `entity_name` 进一步约束根实体名称。 |
| `construct`       | 构造低维实体           |                                |
| `uniform_refine`  | 均匀加密网格           | 当前为占位接口，尚未实现。                  |
| `add_plot`        | 获取绘图接口           | 返回 `MeshPloter`。               |

> [!NOTE]
> 代码文件 `fealpy/mesh/view/mesh.py`

### 2.2 面向旧模块的兼容性视图 FEALPyMesh

> [!NOTE]
> 该视图用于模拟 FEALPy 旧网格（但也包括前面的网格视图接口）。不受实体/非实体语义控制。

| 函数                         | 功能           | 备注                                                |
| -------------------------- | ------------ | ------------------------------------------------- |
| `entity`                   | 获取指定实体的索引/位置 | 与 `entities` 类似，但返回单个张量；`cell/face/edge/node` 可用。 |
| `cell`                     | 获取单元顶点全局编号   | 等价于 `entity('cell')`。                             |
| `face`                     | 获取面顶点全局编号    | 等价于 `entity('face')`。                             |
| `edge`                     | 获取边顶点全局编号    | 等价于 `entity('edge')`。                             |
| `node`                     | 获取节点坐标       | 等价于 `entity('node')`。                             |
| `localEdge`                | 获取单元局部边编号    | 旧 FEALPy 风格属性，由单元 schema 的局部实体关系生成。               |
| `localFace`                | 获取单元局部面编号    | 旧 FEALPy 风格属性，由单元 schema 的局部实体关系生成。               |
| `entity_view`              | 获取单个实体族视图    | 若同一维度存在多个实体族，则抛出异常。                               |
| `entity_barycenter`        | 计算指定实体的重心    | 支持按实体类型和索引切片。                                     |
| `shape_function`           | 计算单元形函数值     | 作用于单元视图；`cell_shape_function` 是同义别名。              |
| `face_shape_function`      | 计算面形函数值      | 作用于面视图。                                           |
| `edge_shape_function`      | 计算边形函数值      | 作用于边视图。                                           |
| `grad_shape_function`      | 计算单元形函数梯度    | 作用于单元视图。                                          |
| `number_of_cells`          | 获取单元数量       | 与 `cell` 视图大小一致。                                  |
| `number_of_faces`          | 获取面数量        | 与 `face` 视图大小一致。                                  |
| `number_of_edges`          | 获取边数量        | 与 `edge` 视图大小一致。                                  |
| `number_of_nodes`          | 获取节点数量       | 直接返回位置数组第一维长度。                                    |
| `number_of_global_ipoints` | 获取全局插值点总数    | 按所有 sector 汇总。                                    |
| `number_of_local_ipoints`  | 获取局部插值点数量    | 默认按 `cell` 计算。                                    |
| `multi_index_matrix`       | 获取多重指标矩阵     | 直接委派给对应实体视图。                                      |
| `quadrature_formula`       | 获取积分公式       | 直接委派给对应实体视图。                                      |
| `cell_to_face`             | 获取单元到面的拓扑映射  | 返回目标实体索引。                                         |
| `cell_to_edge`             | 获取单元到边的拓扑映射  | 返回目标实体索引。                                         |
| `face_to_edge`             | 获取面到边的拓扑映射   | 返回目标实体索引。                                         |
| `boundary_cell_flag`       | 获取单元边界标记     | 返回布尔掩码。                                           |
| `boundary_face_flag`       | 获取边界面标记      | 返回布尔掩码。                                           |
| `boundary_edge_flag`       | 获取边界边标记      | 返回布尔掩码。                                           |
| `boundary_node_flag`       | 获取边界节点标记     | 返回布尔掩码。                                           |
| `boundary_cell_index`      | 获取边界单元索引     | 返回边界实体下标。                                         |
| `boundary_face_index`      | 获取边界面索引      | 返回边界实体下标。                                         |
| `boundary_edge_index`      | 获取边界边索引      | 返回边界实体下标。                                         |
| `boundary_node_index`      | 获取边界节点索引     | 返回边界实体下标。                                         |
| `cell_to_edge_sign`        | 获取单元到边的方向符号  | 当前实现中仍保留在兼容层。                                     |
| `face_to_edge_sign`        | 获取面到边的方向符号   | 当前实现中仍保留在兼容层。                                     |
| `cell_to_ipoint`           | 获取单元到插值点的映射  | 返回插值点索引张量。                                        |
| `face_to_ipoint`           | 获取面到插值点的映射   | 返回插值点索引张量。                                        |
| `interpolation_points`     | 获取插值点坐标      | 可按实体类型批量获取。                                       |
| `bc_to_point`              | 重心坐标转笛卡尔坐标   | 支持分量元组输入。                                         |
| `entity_measure`           | 获取指定实体的测度    | 节点返回零。                                            |
| `edge_tangent`             | 获取边切向量       | 作用于边视图。                                           |
| `edge_unit_tangent`        | 获取边单位切向量     | 对切向量做归一化。                                         |
| `error`                    | 计算两个函数之间的误差  | 委派给单元视图。                                          |
| `face_normal`              | 获取面法向量       | 作用于面视图。                                           |
| `face_unit_normal`         | 获取面单位法向量     | 对法向量做归一化。                                         |
| `grad_lambda`              | 获取重心坐标梯度     | 支持按拓扑维数选择实体视图。                                    |
| `grad_face_lambda`         | 获取面重心坐标梯度    | `grad_lambda(..., TD=top_dimension()-1)` 的封装。     |

> [!NOTE]
> 代码文件 `fealpy/mesh/view/fealpy_api.py`
