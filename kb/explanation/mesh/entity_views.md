# 实体视图

> 版本：v0.1
> 状态：*Draft*
> 入库位置：`kb/explanation/mesh/entity_views.md`

## 一、指定一种实体

在网格视图下，我们通过 `Entity() / Entities()` 方法获得实体视图，通过 `entity` 方法获得实体中的顶点索引或节点坐标。

网格中可能包含不同拓扑维数的实体，而同一维数下又有不同类型的实体。需要在视图中指定实体类型时，一般有以下 4 种格式。

> [!IMPORTANT]
> 同一维数下的实体的序号，就取决于它们被构造的先后顺序；一般而言，它与 OFace 中的顺序相同。

**（1）拓扑维数**

```python
f2_view = mesh.Entity(2) # 拓扑维数是 2 的实体的视图
cell_view = mesh.Entity(-1) # 拓扑维数最高的实体的视图
face_view = mesh.Entity(-2, 1) # 拓扑维数次高的、该维数下的第 1 个的实体的视图
faces = mesh.Entities(-2) # 所有次高维的实体视图的列表
edge_indices = mesh.entity(1) # 拓扑维数是 1 的实体的顶点全局编号
quad_indices = prism_mesh.entity(2, 1) # 在三棱柱中获取四边形的顶点全局编号
```

**（2）拓扑维数类型**

```python
cell_view = mesh.Entity("cell", 0) # 仍然可以传入序号
node_view = mesh.Entity("node")
face_indices = mesh.entity("face")
node_position = mesh.entity("node") # 在 entity 函数中，node 或 0 返回节点坐标
```

序号未定时，默认返回第零个；`Entities` 不再接受序号。

**（3）带序号的拓扑维数类型**

```python
cell_view = mesh.Entity("cell:1")
node_view = mesh.Entity("node:0")
```

**（4）格式名称**

```python
tet = mesh.Entity("tet")
hex_indices = mesh.entity("hex")
```

## 二、视图的使用

视图的接口详见 [kb/design/mesh/mesh_interface](kb/design/mesh/mesh_interface)。首先初始化一个视图：

```python
cell = mesh.Entity("cell")
```

计算单元面积（其它量的计算类似）：

```python
cm = cell.measure(index=...)
```

把笛卡尔坐标函数变成单元上的重心坐标函数：

```python
@cell.barycentric
def my_func(p: Tensor):
    ...
    return ...
```

获取拓扑关系：

```python
cell.to("edge").as_array() # .as_coo() 则拿到 COO 矩阵
```

在实体上存储属性：

```python
cell.set_attribute("value", array)
```
