---
title: 网格数据结构基础
permalink: /docs/zh/mesh-base
key: docs-mesh-base-zh
---


$\quad$ 在偏微分方程数值计算程序设计中， **网格(mesh)**是最核心的数据结构， 
是下一步实现数值离散方法的基础. FEALPy 中核心网格数据结构是用**数组**表示的.

$\quad$ 常见的三角形、四边形、四面体和六面体网格，因为每个单元顶点的个数固定，因此可以
用**节点坐标数组(node)** 和**单元顶点数组(cell)** 来表示，这是一种以**单元为中心的数据结构**.

$\quad$ 如可以用下面的两个 Numpy 数组表示一个包含 4 个节点和 2 个单元的三角形网格.

```python
import numpy as np
node = np.array()
cell = np.array()
```

$\quad$ 很多时候，我们还需要获得**边数组(edge)**和**面数组(face)**, 其实它们都可以由
cell 数组快速生成. 下面我们以三角形网格为例, 来详细介绍 edge 数组的生成算法.

```python
# 介绍生成算法
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

大多数情况下, 一个网格的几何维数和拓扑维数是相同的, 但也可以不一样,
如三维空间中的一条曲线离散的一维网格, 它
* `GD == 3`
* `TD == 1`

