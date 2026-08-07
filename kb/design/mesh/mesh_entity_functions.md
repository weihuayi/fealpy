# 网格实体函数

> [!WARNING]
> 纯算法：仅承载算法功能本身，完全与数学过程对应，供计算视图调取。不作为用户接口，不做数学算法承诺范围以外的函数分配、资源调度、入参解析和输出后处理。

## 一、标准函数表

> [!WARNING]
> 标准函数表应具有普适性。它规范在多种形状格式中出现的函数的签名，避免函数语义不一致。因此只有在多种实体中出现的函数才能进入实体函数的标准函数表。

> [!WARNING]
> `EntitySchema` 类是规定标准函数签名的唯一对象，子类不得定义标准函数签名。非标准函数不受影响。

> [!WARNING]
> `EntitySchema` 类中的函数实现应是最终实现（面向所有实体），子类不得覆盖。如果出现需了要覆盖的情形，则该函数实现不应在基类中。

> [!IMPORTANT]
> 基类 `EntitySchema` 继承到最终实体类之间，至多只能存在一个中间类。平衡代码复用程度和维护难度。

| 分类  | 函数                                | 功能                   |
| --- | --------------------------------- | -------------------- |
| 拓扑  | `boundary`                        | 获得边界信息               |
| 拓扑  | `local_entity`                    | 局部子实体编号              |
| 拓扑  | `relation`                        | 获取拓扑关系               |
| 拓扑  | `size`                            | 实体数量                 |
| 插值点 | `multi_index`                     | 获取多重指标，0-轴广播形式/张量积形式 |
| 插值点 | `num_multi_index`                 | 计算多重指标数量             |
| 插值点 | `global_permutations`             | 子实体局部朝向到全局朝向的顶点置换矩阵  |
| 插值点 | `vo_to_do`                        | 从顶点置换矩阵获得多重指标置换矩阵    |
| 几何  | `barycenter`                      | 单元重心                 |
| 几何  | `barycentric`                     | 装饰器，把笛卡尔坐标函数变为重心坐标函数 |
| 几何  | `bc_to_point`                     | 重心坐标转换为笛卡尔坐标         |
| 几何  | `geo_dimension`                   | 几何维数                 |
| 几何  | `grad_shape_function_barycentric` | 形函数对重心坐标的导数          |
| 几何  | `grad_shape_function_cartesian`   | 形函数对笛卡尔坐标的导数         |
| 几何  | `grad_shape_function_reference`   | 形函数对参考坐标的导数          |
| 几何  | `integral`                        | 计算输入函数的数值积分          |
| 几何  | `jacobi_matrix`                   | 雅可比矩阵，物理坐标关于参考坐标的梯度  |
| 几何  | `measure`                         | 单元测度                 |
| 几何  | `normal`                          | 单元法向                 |
| 几何  | `quadrature_formula`              | 获取积分公式               |
| 几何  | `shape_function`                  | 计算形函数值               |
| 几何  | `tangent`                         | 单元切向                 |

> [!NOTE]
> 代码文件 `fealpy/mesh/schema/entity_schema.py`
> 这些函数签名按分类和字母表顺序，放在基类中。

## 二、特殊函数
