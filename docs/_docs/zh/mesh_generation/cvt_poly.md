---
title: CVT多边形网格生成
permalink: /docs/zh/mesh_generation/cvt_poly
key: docs-cvtpoly-zh
---

# 网格介绍

<!--
介绍voronoi网格和CVT网格的一些性质，说明Fealpy中的CVT多边形网格基于CVTPMesher
生成，以正方形为例给出一个基本的例子。
-->

Voronoi网格剖分是一种多边形网格剖分. 设 $\Omega\subset\mathbb{R}^d$ 是一个开区
域, 并且 $\mathcal{Z} =\{\boldsymbol{z_i}\}_{i=0}^{N-1}\subset\Omega$ 是一组给定的点. 一个 
Voronoi 剖分 $\mathcal{V} = \{V_i\}_{i=0}^{N-1}$ 是 $\Omega$ 的一种多边形网格剖分, 它的
每个区域 $V_i$ 定义为
$$    
    V_i=\{\boldsymbol{x}\in\Omega:|\boldsymbol{x}-\boldsymbol{z_i}|<|\boldsymbol{x}-\boldsymbol{z_j}|,\text{其中}j\neq i\}
$$

称 $V_i$ 为一个 Voronoi 单元, 点集 $\mathcal{Z}$ 中的点被称为生成子(Generator),  CVT 是
Vorionoi 剖分的一种特殊剖分, 当每个 Voronoi 单元的生成子也是其质心时, 该 Voronoi 剖
分称为一个 CVT.

在Fealpy中，我们可以通过mesh中的CVTPMesher来生成CVT多边形网格，首先需要利用HalfEdgeMesh2d中的
from_edges建立初始网格区域，再利用CVTPMesher生成初始Voronoi网格区域，再利用VoroAlgorithm中的方法
进行优化得到最终的CVT网格。

```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d, CVTPMesher,VoroAlgorithm

from scipy.spatial import Voronoi, voronoi_plot_2d

nodes = np.array([
    ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
facets = np.array([
    (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
subdomain = np.array([
    (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
voromesher = CVTPMesher(mesh)
vor = voromesher.voronoi_meshing(nb=2)

i =0
avoro = VoroAlgorithm(voromesher)
while i<100:
    vor = avoro.lloyd_opt(vor)
    i+=1

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, color='k', showindex=False)
mesh.find_node(axes, node=vor.points, showindex=True)
voronoi_plot_2d(vor, ax=axes,show_vertices = False,point_size = 0.3)
plt.show()
```

<img src='../../../assets/images/mesh-generation/cvt-poly/square.png' width='350'  title = '正方形区域'>

<!--
简单介绍一下生成网格所需的数据结构以及各个步骤的作用
-->

# 网格生成
引入密度函数，密度函数的区间为(0,1]，越接近0密度越小，当为1时密度最大。若给定的
密度区间不为(0,1]，则需要先手动做等价的替换，将区间变换到(0,1]再进行传入。

网格生成由三部分组成:边界加密，生成初始voronoi网格，优化为CVT网格

## 边界加密与重构边界
采用一致加密，但对于不同边长不同密度可以自适应加密。

采用VoroCrust算法——一种采用镜像原理的方法重构边界
## 生成初始voronoi网格
### 随机布点
采用随机布点方法，在区域内随机生成一组点，判断离边界点的距离，当大于一定值时则留下，否则抛弃。

同时还可引入密度影响，对于随机生成的点，再在(0,1)间随机生成一组随机数，算出该随机点的密度，
将密度将对应的随机数进行比较，若大于随机数则留下来，若小于则丢掉。
### 其他布点方法
除随机布点外，其他还有背景网格布点法等，待算法成熟再介绍加进去

## 优化为CVT网格
将初始的voronoi网格优化为CVT网格
### Lloyd's优化方法
Lloyd's 算法是生成CVT网格的常用方法, 即先固定生成子集 $\boldsymbol{z}$, 然后对区域进行
Voronoi 剖分, 生成 Voronoi 网格剖分 $\mathcal{V}(\boldsymbol{z})$, 再固定 
$\mathcal{V}$, 然后将 $\boldsymbol{z}$ 中的点移动到各自区域的质心.

给定区域 $V_i$, 定义其上的密度函数 $\rho$, 则 $V_i$ 的质心 $z_i^{*}$ 定义为:
$$
z_i^{*} = \left(\int_{V_i}\rho(x)\text{d}x\right)^{-1}\int_{V_i}x\rho(x)\text{d}x
$$

关于近似积分，我仍然存在问题，明天我会再找老师讨论一下。
### 其他优化方法
Lloyd's迭代是局部算法，我们还可以采用全局优化算法，写完密度函数后会编写全局优化算法的函数

# 网格示例

给出各种网格区域的示例。



