---
title: 网格工厂
permalink: /docs/zh/mesh/mesh-factory
key: docs-mesh-factory-zh
author: wpx
---

​			FEALPy中的网格工厂模块是用来快速生成各种类型的常用网格，方便用户学习和使用FEALPy
# one_triangle_mesh
参数说明:

- meshtype：网格形状，直角三角形`(iso)`，等边三角形`(equ)`



# one_quad_mesh

参数说明:

- meshtype：网格形状，square，rectangle，rhombus，



<div align="center">
    	<img src='../../../assets/images/mesh/mesh-base/quad.png' width="300"> 
    	<img src='../../../assets/images/mesh/mesh-base/quad.png' width="300"> 
</div>

<center style="color:#C0C0C0;text-decoration:underline">图1.xxx.jpg</center>

# one_quad_mesh

# boxmesh2d

用来快速生成二维矩形结构网格

参数说明:

- box：矩形网格的位置

- nx：x方向剖分段数

- ny：y方向剖分段数

- meshtype：网格剖分类型，包含三角形`('tri')`，四边形`('quad')`，三角形对偶的多边形`('poly')`。

使用方法如下

```python
from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib.pyplot as plt

box = [0,1,0,1]
mesh = MF.boxmesh2d(box=box,nx=2,ny=2,meshtype='tri')
# 画图
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
```

# interval_mesh
用来快速生成一维线网格
参数说明:

- interval：线的位置
- nx：x方向剖分段数
# boxmesh3d

用来快速生成三维矩形结构网格
参数说明:

- box：矩形网格的位置
- nx：x方向剖分段数
- ny：y方向剖分段数
- nz：z方向剖分段数
- meshtype：网格剖分类型，包含四面体`('tet')`，六面体`('hex')`。

使用方法如下

```python
from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib.pyplot as plt

box = [0,1,0,1,0,1]
mesh = MF.boxmesh3d(box=box,nx=2,ny=2,meshtype='hex')
# 画图
fig = plt.figure()
axes = fig.gca(projection='3d')
mesh.add_plot(axes)
plt.show()
```

# spcial_boxmesh2d
用来快速生成二位矩形特殊网格
参数说明:

- box：矩形网格的位置
- n ：两个方向上的剖分段数
- meshtype：网格剖分类型，包含鱼骨形`('fishbone')`，十字形`('cross')`，米字形`('rice')`，非一致网格`('nonuniform')`。

```python
from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib.pyplot as plt

box = [0,1,0,1,0,1]
mesh = MF.boxmesh3d(box=box,nx=2,ny=2,meshtype='hex')
# 画图
fig = plt.figure()
axes = fig.gca(projection='3d')
mesh.add_plot(axes)
plt.show()
```

# lshape_mesh
参数说明:

- n ：加密次数

# unitcirclemesh
 利用 distmesh 算法生成单位圆上的非结构三角形或多边形网格

# triangle
生成矩形区域上的非结构网格网格

# delete_cell
利用 threshhold 来删除一部分网格单元。threshold 以单元的重心为输入参数，返回一个逻辑数组，需要删除的单元标记为真。

参数说明:

- node ：
- cell：
- threshold：