# 网格拓扑关系

> 版本：v0.1
> 状态：*Draft*
> 入库位置：`kb/explanation/mesh/topology_realation.md`

## 一、拓扑关系对象

> [!NOTE]
> 代码文件 `fealpy\mesh\storage\relation.py`

拓扑关系接口总是返回拓扑关系对象 `Relation`，该对象具有以下属性：

| 属性              | 类型             | 含义          |
| --------------- | -------------- | ----------- |
| `src_name`      | str            | 映射关系源的实体名称  |
| `tgt_name`      | str            | 映射关系目标的实体名称 |
| `tgt_indices`   | Tensor         | 目标实体索引      |
| `src_indices`   | Tensor \| None | 源实体索引       |
| `local_indices` | Tensor \| None | 目标在源中的局部编号  |

### 1.1 布局

针对不同的拓扑关系，Relation 有两种数据布局方式：

**（1）同质性布局**（稠密）。只有当每个源实体映射到相同数量的目标实体的情形时，才可使用这种方式；此时

- `tgt_indices` 为 2 维张量；
- `src_indices` 为 None。

**（2）异质性布局**（稀疏）。这种方式适用于任意关系，此时 `tgt_indices` 和 `src_indices`

- 均为 1 维张量；
- 长度相等，都等于映射关系的数量。

### 1.2 视图与操作

用户可以将 Relation 转化成稠密张量或稀疏矩阵以便参与后续计算。

```python
relation.as_array() # 仅同质性布局可用
relation.as_coo()
relation.as_csr()
```

`inverse` 方法可以方便地逆向拓扑关系，就像把 cell_to_face 变成 face_to_cell。

```python
inv_relation = relation.inverse()
```

> [!IMPORTANT]
> 经 `inverse` 逆向以后的拓扑关系总是异质性布局的。

## 二、使用拓扑关系接口

实体视图和兼容性网格视图中均提供了直观的拓扑关系接口。结合实体视图，我们一般可以这样使用：

**（1）指定实体形状名字**

```python
tet2tri = mesh.Entity("tet").to("tri").as_array()
```

**（2）指定实体维数**

```python
prism2quad = mesh.Entity("cell").to("face:1").as_array()
# 或者
prism2quad = mesh.Entity("cell").to(-2, 1).as_array()
```

这两种 `to` 中的传参规则与 `mesh.Entity` 的是完全一样的，而且 `to` 和 `Entity` 两处的传参格式可以不同。

**（3）传入目标实体的视图**

```python
edge = mesh.Entity("edge")
cell = mesh.Entity("cell")
cell2edge = cell.to(edge).as_array()
```

> [!NOTE]
> 惰性计算：除了构造网格时产生的拓扑关系，其它拓扑关系会在第一次访问时被自动推导出来。

## 三、拓扑计算器

> [!INFO] 代码文件 `fealpy\mesh\topology\builder.py`

| 工具                             | 用途                                       |
| ------------------------------ | ---------------------------------------- |
| `TopologyBuilder.construct`    | 构造：从最高维实体构造低维实体，同时产出最高维实体到所有生成的低维实体的拓扑关系 |
| `TopRelationConnector.connect` | 连接：计算已经存在的两种实体之间、高维到低维的拓扑关系              |
| `TopRelationInferer.infer`     | 推导：从两个拓扑关系的复合得到新的拓扑关系                    |

拓扑关系查询、计算规则：

1. 如果已经存在于 `block.relations` 中，则直接返回；
2. 如果存在逆向关系，则求逆后返回；
3. 如果起点维数比终点高，尝试用连接的方式计算；
4. 最后查找中间维数实体，尝试用推导的方式计算；
5. 如果以上都不行，抛出 `ValueError`。

> [!Caution]
> 以上逻辑并未探索所有可行的计算路径。

> [!WARNING]
> `to` 只能对已存在的实体计算拓扑关系，理论上不应自动构造实体。
