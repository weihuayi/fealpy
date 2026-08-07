# FEALPy | 网格数据结构与 Mesh 模块架构设计

| 项目   | 内容                       |
| ---- | ------------------------ |
| 文档类型 | 架构设计说明                   |
| 模块   | Mesh Core Infrastructure |
| 状态   | *Active*                 |
| 目标版本 | 下一代网格系统                  |
| 作者   | Albert                   |
| 更新时间 | 2026-04-20               |

## 一、文档目的

本文档给出下一代 Mesh 数据结构的整体架构设计，用于：

- 统一团队对 Mesh 核心结构的理解
- 支撑后续代码实现与模块划分
- 为未来扩展提供清晰边界

本设计重点关注：

- 架构分层
- 数据结构
- 接口职责
- 内存布局

本文档 **不定义具体算法实现、排序规则或去重策略**，这些属于算法模块的职责。

## 二、背景与设计问题

有限元软件中的网格结构面临一个典型矛盾：

数值计算希望数据结构具有：张量友好、内存连续、易于批量计算；
而网格拓扑本质上：非同质、多维实体混合、拓扑关系复杂。

传统网格软件通常使用：

- Half-edge
- Winged-edge
- DAG
- Pointer graph

这些结构非常适合网格编辑和拓扑修改，但并不适合：

- 数值积分
- 批量算子组装
- GPU计算

因此，本设计采用：**张量化存储 + 拓扑关系显式记录** 的结构。

## 三、总体架构

Mesh 系统由三个层次组成：

- 规则模板层（EntitySchema）
- 拓扑存储层（MeshBlock / EntitySector）
- 计算视图层（Mesh / EntityView）

架构关系如下：
```
User Code
    │
    ▼
Mesh / EntityView
    │
    ▼
MeshBlock / EntitySector
    │
    ▼
EntitySchema
```

各层职责如下：

| 层     | 职责            |
| ----- | ------------- |
| 规则模板层 | 定义实体规则与形状操作   |
| 拓扑存储层 | 保存 mesh 的离散事实 |
| 计算视图层 | 提供用户 API      |

这种结构将系统中的三个核心概念分离：

| 概念 | 含义 |
|----|----|
| 规则 | 拓扑模板与实体定义 |
| 事实 | 具体 mesh 数据 |
| 操作 | 计算与构造算法 |

## 四、规则模板层

### 4.1 设计目标

规则模板层针对每一种实体形状，制定拓扑与几何操作规则。

每一种形状都拥有其对应的模板，称 EntitySchema；它只定义规则，不包含任何 mesh 实例数据。

### 4.2 Entity Schema

`EntitySchema` 表示一种 **实体类型的规则集合**，其典型信息包括：

| 属性          | 含义        |
| ----------- | --------- |
| name        | 实体名称      |
| top_dim     | 拓扑维数      |
| local_faces | 子实体局部模板字典 |

例如四棱锥（金字塔） Scheme 中记录：

- 拓扑维数：3
- `local_faces`：`{'quad': [[0, 1, 2, 3]], 'tri': [[0, 1, 4], [2, 3, 4], [0,2 , 4], [1, 3, 4]]}`
- 形状相关操作，如
	- measure
	- barycenter
	- normal
	- tangent

这些操作依赖实体索引和节点坐标，但 Schema 自身 **不持有这些数据**。

### 4.3 Schema Registry

系统通过注册表维护所有实体类型。例如：

- node
- edge
- tri
- quad
- tet
- prism
- pyramid
- hex

注册表用于：

- 根据实体名称字符串查找 Schema 类
- 在拓扑构造阶段创建实体

## 五、拓扑存储层

### 5.1 设计目标

拓扑存储层负责保存 mesh 的所有 **离散事实**。包括：

- 节点坐标
- 实体连接关系
- 实体之间的拓扑关系

Storage 层 **不包含任何推导逻辑**。

### 5.2 Mesh Storage

`MeshBlock` 类型对象是 Mesh 的核心数据容器。主要字段包括：

| 字段            | 含义       | 类型                                    |
| ------------- | -------- | ------------------------------------- |
| positions     | 节点坐标张量   | Tensor                                |
| blocks        | 各类实体的数据块 | dict[str, EntityBlock]                |
| relations     | 实体之间关系   | dict[tuple[str, str], EntityRelation] |
| root_entities | 顶层实体名称   | list[str]                             |

> [!WARNING] 具体字段因存储格式而异
> 本文档仅以连接关系 `(N_entity, N_node_per_entity)` 存储的网格实体为例。对于半边网格、组合映射网格等，可能具有不同的 Storage 和 Schema 类型。

### 5.3 Entity Block

`EntityBlock` 类型对象用于表示某个网格中 **一批同类型实体**。典型例子：

- 某一网格中的所有三角形
- 某一网格中的所有四面体
- 某一网格中的所有边

例如三棱柱网格有这些 Blocks：

| 实体    | 形状          |
| ----- | ----------- |
| node  | (Nnode, 1)  |
| edge  | (Nedge, 2)  |
| tri   | (Ntri, 3)   |
| quad  | (Nquad, 4)  |
| prism | (Nprism, 4) |

EntityBlock 的职责：

- 存储实体连接关系
- 关联 Schema
- 提供基本元信息

### 5.4 Entity Relation

`EntityRelation` 表示实体之间的拓扑关系。例如：

- cell → face
- face → edge
- edge → node

典型关系：

| 关系 | 结构 |
|----|----|
| cell_to_face | (NC, Nface_per_cell) |
| cell_to_edge | (NC, Nedge_per_cell) |
| face_to_node | (NF, Nnode_per_face) |

Relation 只存储：

- 索引映射
- 局部编号信息

Relation 不负责：

- 构造逻辑
- 去重策略

## 六、计算视图层

### 6.1 设计目标

计算视图层提供用户 API 的唯一入口。用户不需要直接操作 Storage。针对网格视图 `Mesh` 和实体视图 `EntityView`，我们约定：

**1. 视图是接口**

视图不是算法实现的载体，不能在视图类中实现数学算法。

**2. 视图是外壳**

视图在初始化时只能做属性赋值。

### 6.2 Mesh API

`Mesh` 类型对象是用户的主要入口。典型接口包括：

- 获取实体
- 查询维数
- 枚举实体

Mesh API 的职责：

- 封装 Block
- 提供统一访问入口

### 6.3 Entity View

`EntityView` 类型对象表示某一类实体的视图。例如：

- 所有三角形
- 所有边

`EntityView` 提供操作：

- measure
- barycenter
- normal
- tangent

这些操作内部通过 Schema 实现。

## 七、拓扑构建算法

### 7.1 设计目标

拓扑构建负责构造实体和实体之间的关系。例如：

- tri → edge
- edge → node

Builder 属于算法模块；Storage 不负责这些逻辑。
具体去重规则与排序规则 **不属于本架构文档范围**。

## 八、内存布局规范

为了保证数值计算效率，Mesh 数据必须采用 **张量友好布局**。

### 8.1 坐标数组

节点坐标存储为：

```
positions : (Nnode, GD)
```

其中：

| 符号 | 含义 |
|----|----|
| Nnode | 节点数量 |
| GD | 几何维数 |

### 8.2 实体连接数组

实体连接统一采用二维数组：

```
(N_entity, N_vertex)
```

例如：

| 实体 | Layout |
|----|----|
| edge | (Nedge, 2) |
| tri | (Ntri, 3) |
| quad | (Nquad, 4) |
| tet | (Ntet, 4) |
| hex | (Nhex, 8) |

### 8.3 关系数组

实体关系通常采用：

```
(N_src, N_local_target)
```

例如：

```
cell_to_face
cell_to_edge
```

这种布局具有：

- 内存连续
- GPU友好
- 向量化友好

## 九、模块结构

```text
mesh/
	#---------- 核心骨架 ----------#
	schema/                   # 形状格式 | 存放实体算法
		classic/                  # 经典低阶形状
			...
		entity_schema.py          # 基类
		registry.py               # 提供 名字-形状类 映射
		utils.py                  # 工具函数
	
	stoarge/                  # 网格存储 | 存放网格数据结构
		mesh_storage.py           # 网格数据结构
		relation.py               # 拓扑关系存储结构
	
	view/                     # 计算视图 | 定义用户接口
		entity_view.py            # 实体视图
		mesh.py                   # 网格视图
		fealpy_api.py             # 老接口兼容性视图
	
	#---------- 算法类 ----------#	
	topology/                 # 算法：拓扑构造、推断
		boundary.py               # 边界实体推断
		builder.py                # 网格构造（construct）算法、拓扑关系推导
	
	ipoints.py                # 算法：插值点位置、全局编号、全局映射
	join.py                   # 算法：简单合并网格
	merge.py                  # 算法：按距离合并重复点
	transform.py              # 算法：参考单元到物理单元变换
	uniform_refine.py         # 算法：均匀加密
	
	#---------- 工具类 ----------#
	plotting/                 # 工具：基于 Matplotlib 的网格绘图
		artist.py
		classic.py
	
	vtk_writter.py            # 工具：VTK 存储
	vtk_reader.py             # 工具：VTK 读取
	
	#---------- 兼容层 ----------#
	factory.py                # 提供 TriangleMesh 等类
	mesh_base.py              # 提供 SimplexMesh 等类
```

> [!NOTE]
> 形成 Mesh 模块核心骨架的，是 `schema/`、`storage/` 和 `view/`。

**重要文档**：

- [实体函数表](kb/design/mesh/mesh_entity_functions.md)
- [计算视图接口表](kb/design/mesh/mesh_interface)

## 十、总结

本架构的核心思想是：

**规则、事实、操作三者分离。**

具体表现为：

- Schema 定义规则
- Storage 保存事实
- Builder 实现算法
- View 提供接口

这种设计具有以下优势：

- 数据结构清晰
- 计算友好
- 扩展性强
- 易于维护

同时为未来扩展提供基础，例如：

- 混合网格
- GPU计算
- 自适应网格
- 多物理场耦合

该架构将作为下一代有限元基础设施的重要组成部分。