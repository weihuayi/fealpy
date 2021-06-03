"""
Notes
-----
这是一个展示 MeshFactory 对象使用方法的脚本

Author 
------
Huayi Wei <weihuayi@xtu.edu.cn>

Date
----
2021.06.03 08:48:52
"""

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF

box2d = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box2d, nx=10, ny=10, meshtype='tri')
mesh = MF.boxmesh2d(box2d, nx=10, ny=10, meshtype='quad')
mesh = MF.boxmesh2d(box2d, nx=10, ny=10, meshtype='poly')
mesh = MF.special_boxmesh2d(box2d, n=10, 
        meshtype='fishbone')
mesh = MF.special_boxmesh2d(box2d, n=10, 
        meshtype='rice')
mesh = MF.special_boxmesh2d(box2d, n=10, 
        meshtype='cross')
mesh = MF.special_boxmesh2d(box2d, n=10, 
        meshtype='nonuniform')
mesh = MF.unitcirclemesh(0.1, meshtype='poly')
mesh = MF.triangle(box2d, 0.1, meshtype='poly')

fig = plt.figure()
axes= fig.gca()
mesh.add_plot(axes)
plt.show()




