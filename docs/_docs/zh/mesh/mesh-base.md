---
title: 网格数据结构基础
permalink: /docs/zh/mesh/mesh-base
key: docs-mesh-base-zh
---


$\quad$ 在偏微分方程数值计算程序设计中， **网格(mesh)**是最核心的数据结构， 
是下一步实现数值离散方法的基础. FEALPy 中核心网格数据结构是用**数组**表示的.

$\quad$ 常见的三角形、四边形、四面体和六面体网格，因为每个单元顶点的个数固定，因此可以
用**节点坐标数组(node)** 和**单元顶点数组(cell)** 来表示，这是一种以**单元为中心的数据结构**.

$\quad$ 如可以用下面的两个 Numpy 数组表示一个包含 4 个节点和 2 个单元的三角形网格.

```python
import numpy as np
node = np.array([[0.0, 0.0],[1.0, 0.0],[1.0, 1.0],[0.0, 1.0]],dtype=np.float)
cell = np.array([[1,2,0],[3,0,2]],dtype=np.int)
```

$\quad$ 很多时候，我们还需要获得**边数组(edge)**和**面数组(face)**, 其实它们都可以由
cell 数组快速生成. 下面我们以三角形网格为例, 来详细介绍 edge 数组的生成算法.

```python
    NEC = 3
    localEdge = np.array([(1, 2), (2, 0), (0, 1)], dtype=np.int)

    totalEdge = cell[:, localEdge].reshape(-1, 2)
    stotalEdge = np.sort(totalEdge, axis=1)
    _, i0, j = np.unique(stotalEdge, return_index=True, return_inverse=True, axis=0)

    edge = totalEdge[i0]
    NE = i0.shape[0]

    i1 = np.zeros(NE, dtype=np.int)
    NC = cell.shape[0]
    i1[j] = np.arange(3*NC)

    // 边与单元的拓扑关系数组
    edge2cell = np.zeros((NE, 4), dtype=np.int)
    t0 = i0//3
    t1 = i1//3
    k0 = i0%3
    k1 = i1%3
    edge2cell[:, 0] = t0
    edge2cell[:, 1] = t1
    edge2cell[:, 2] = k0
    edge2cell[:, 3] = k1
```

注意, 三维网格的 face 数组的生成算法和上面 edge 数组的生成算法本质上是完全一样的.

$\quad$ FEALPy 把 node、edge、face 和 cell 统称为网格中的**实体(entity)**. FEALPy 约定
node 和 cell 分别是网格中的**最低维**和**最高维**实体, 另外还约定

* 在一维情形下，edge、face 和 cell 的意义是相同的.
* 在二维情形下，edge 和 face 意义是相同的.

$\quad$ FEALPy 除了上面的约定外, 还约定了一些常用变量名称的意义, 如 


|变量名 | 含义 |
|:--- | :----|
|GD	| 网格的几何维数 |
|TD	| 网格的拓扑维数 |
|NN	| 网格中 node 的个数 |
|NE	| 网格中 edge 的个数 |
|NF	| 网格中 face 的个数 |
|NC	| 网格中 cell 的个数 |
|NVC | 网格中每个 cell 包含的点的个数 |
|NEC | 网格中每个 cell 包含的边的个数 |
|NVF | 网格中每个 face 包含的点的个数 |
|NEF | 网格中每个 face 包含的边的个数 |
|node| 节点数组，形状为 (NN, GD) |
|edge| 边数组，形状为 (NE, 2) |
|face| 面数组，形状为 (NF, NVF) |
|cell| 单元数组，形状为 (NC, NVC) |
|node2cell | 节点与单元的相邻关系 |
|edge2cell | 边与单元的相邻关系 |
|cell2edge | 单元与边的相邻关系 |

$\quad$大多数情况下, 一个网格的几何维数和拓扑维数是相同的, 但也可以不一样,
如三维空间中的一条曲线离散的一维网格, 它
* `GD == 3`
* `TD == 1`

$\quad$可以通过以下方法来获取网格对象的各种信息

|成员函数名 | 功能 |
|:--- | :----|
|mesh.geo\_dimension() | 获得网格的几何维数 |
|mesh.top\_dimension() | 获得网格的拓扑维数 |
|mesh.number\_of\_nodes() |	获得网格的节点个数 |
|mesh.number\_of\_cells() |	获得网格的单元个数 |
|mesh.number\_of\_edges() |	获得网格的边个数 |
|mesh.number\_of\_faces() |	获得网格的面的个数 |
|mesh.number\_of\_entities(etype) |	获得 etype 类型实体的个数 |
|mesh.entity(etype) | 获得 etype 类型的实体 |
|mesh.entity\_measure(etype) | 获得 etype 类型的实体的测度 |
|mesh.entity\_barycenter(etype) | 获得 etype 类型的实体的重心 |
|mesh.integrator(i) | 获得该网格上的第 i 个积分公式 |

网格对象的常用方法成员（属性）列表。表格中 etype 值可以是 0, 1, 2, 3 或者字符串 
‘cell’, ‘node’, ‘edge’, ‘face’。对于二维网格，etype 的值取 ‘face’ 和 ‘edge’ 是等
价的，但不能取 3。

$\quad$ds中的方法成员也有一些网格对象的信息

|成员函数名 | 功能 |
|:--- | :----|
|cell2cell = mesh.ds.cell\_to\_cell(...) | 单元与单元的邻接关系 |
|cell2face = mesh.ds.cell\_to\_face(...) | 单元与面的邻接关系 |
|cell2edge = mesh.ds.cell\_to\_edge(...) | 单元与边的邻接关系 |
|cell2node = mesh.ds.cell\_to\_node(...) | 单元与节点的邻接关系 |
|face2cell = mesh.ds.face\_to\_cell(...) | 面与单元的邻接关系 |
|face2face = mesh.ds.face\_to\_face(...) | 面与面的邻接关系 |
|face2edge = mesh.ds.face\_to\_edge(...) | 面与边的邻接关系 |
|face2node = mesh.ds.face\_to\_node(...) | 面与节点的邻接关系 |
|edge2cell = mesh.ds.edge\_to\_cell(...) | 边与单元的邻接关系 |
|edge2face = mesh.ds.edge\_to\_face(...) | 边与面的邻接关系 |
|edge2edge = mesh.ds.edge\_to\_edge(...) | 边与边的邻接关系 |
|edge2node = mesh.ds.edge\_to\_node(...) | 边与节点的邻接关系 |
|node2cell = mesh.ds.node\_to\_cell(...) | 节点与单元的邻接关 |
|node2face = mesh.ds.node\_to\_face(...) | 节点与面的邻接关系 |
|node2edge = mesh.ds.node\_to\_edge(...) | 节点与边的邻接关系 |
|node2node = mesh.ds.node\_to\_node(...) | 节点与节点的邻接关系 |

|成员函数名 | 功能 |
|:--- | :----|
|isBdNode = mesh.ds.boundary\_node\_flag() | 一维逻辑数组，标记边界节点 |
|isBdEdge = mesh.ds.boundary\_edge\_flag() | 一维逻辑数组，标记边界边 |
|isBdFace = mesh.ds.boundary\_face\_flag() | 一维逻辑数组，标记边界面 |
|isBdCell = mesh.ds.boundary\_cell\_flag() | 一维逻辑数组，标记边界单元 |
|bdNodeIdx = mesh.ds.boundary\_node\_index() | 一维整数数组，边界节点全局编号 |
|bdEdgeIdx = mesh.ds.boundary\_edge\_index() | 一维整数数组，边界边全局编号 |
|bdFaceIdx = mesh.ds.boundary\_face\_index() | 一维整数数组，边界面全局编号 |
|bdCellIdx = mesh.ds.boundary\_cell\_index() | 一维整数数组，边界单元全局编号 |














