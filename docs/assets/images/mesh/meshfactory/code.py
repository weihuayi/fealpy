#!/usr/bin/python3
'''    	
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年10月30日 星期六 19时26分25秒
'''  
from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib.pyplot as plt

box = [0,1,0,1]
mesh = MF.boxmesh2d(box=box,nx=1,ny=1,returnnc=True)
print(mesh)
'''
# 画图
fig = plt.figure()
axes = fig.gca()
#axes = fig.gca(projection='3d')
mesh.add_plot(axes)
plt.show()
'''
