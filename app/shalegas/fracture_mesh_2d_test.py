
import pickle

import numpy as np
from scipy.spatial import KDTree 

import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory


"""

Notes
-----
用于制作带裂缝的两相流网格模型, 并把模型保存为 pickle 文件 
"""


def is_fracture_cell(mesh):

    NC = mesh.number_of_cells()

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    isFractureCell = np.zeros(NC, dtype=np.bool_)

    def f(x):
        a = x[0]
        b = x[1]
        d = x[2]
        flag0 = (node[cell[:, 0], d[0]] >= b[0]) & (node[cell[:, 0], d[0]] <= b[1]) & (node[cell[:, 0], d[1]] == a)
        flag1 = (node[cell[:, 1], d[0]] >= b[0]) & (node[cell[:, 1], d[0]] <= b[1]) & (node[cell[:, 1], d[1]] == a)
        flag2 = (node[cell[:, 2], d[0]] >= b[0]) & (node[cell[:, 2], d[0]] <= b[1]) & (node[cell[:, 2], d[1]] == a)
        isFractureCell[flag0 | flag1 | flag2] = True

    # fracture
    a = [5, 1, 3, 5, 7, 9]
    b = [(0.5, 9.5), (2, 8), (3, 7), (2, 8), (1, 9), (4, 6)]
    d = [(0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    list(map(f, zip(a, b, d)))

    return isFractureCell

box = [0, 10, 0, 10] # m 
mesh = MeshFactory.boxmesh2d(box, nx=10, ny=10, meshtype='tri')
mesh.box = box

for i in range(14):
    isFractureCell = is_fracture_cell(mesh)
    mesh.bisect(isFractureCell)

# 构建模型和数据参数
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 构建 KDTree，用于定义注入和生成井的网格点位置
node = mesh.entity('node')
tree = KDTree(node)

p0 = np.array([(9.5, 5)], dtype=np.float64) # 生成井位置
p1 = np.array([(0, 0), (0, 10)], dtype=np.float64) # 注水井位置 

_, location0 = tree.query(p0)
_, location1 = tree.query(p1)

print('生产井的位置:', node[location0])
print('注水井的位置:', node[location1])


# 裂缝标签
isFractureCell = is_fracture_cell(mesh)
mesh.celldata['fracture'] = isFractureCell

# 渗透率
mesh.celldata['permeability'] = np.zeros(NC, dtype=np.float64) # 1 d = 9.869 233e-13 m^2
mesh.celldata['permeability'][ isFractureCell] = 6 # 裂缝 
mesh.celldata['permeability'][~isFractureCell] = 2 # 岩石 

# 孔隙度
mesh.celldata['porosity'] = np.zeros(NC, dtype=np.float64) # 百分比
mesh.celldata['porosity'][ isFractureCell] = 0.1 # 裂缝
mesh.celldata['porosity'][~isFractureCell] = 0.3 # 岩石

# 拉梅第一常数
mesh.celldata['lambda'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['lambda'][ isFractureCell] = 0.5e+2 # 裂缝 
mesh.celldata['lambda'][~isFractureCell] = 1.0e+2 # 岩石 

# 拉梅第二常数
mesh.celldata['mu'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['lambda'][ isFractureCell] = 1.5e+2 # 裂缝 
mesh.celldata['lambda'][~isFractureCell] = 3.0e+2 # 岩石 

# Biot 系数
mesh.celldata['biot'] =  np.zeros(NC, dtype=np.float64) # 
mesh.celldata['biot'][ isFractureCell] = 1.0 # 裂缝 
mesh.celldata['biot'][~isFractureCell] = 1.0 # 岩石 

# MPa 固体体积模量 K = lambda + 2*mu/3
mesh.celldata['K'] =  np.zeros(NC, dtype=np.float64) # 
mesh.celldata['K'] =  mesh.celldata['lambda'] + 2*mesh.celldata['mu']/3

# 初始压强
mesh.celldata['pressure'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['pressure'][:] = 3.0 # MPa

# 初始应力
mesh.celldata['stress'] = 2.0e+2 # MPa 初始应力 sigma_0, sigma_eff

# 初始水的饱和度 
mesh.celldata['water saturation'] = np.zeros(NC, dtype=np.float64)
# 初始油的饱和度 
mesh.celldata['oil saturation'] = 1 - mesh.celldata['water saturation'] 

# 油的开采速度
mesh.nodedata['Fo'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['Fo'][location0] = 7.0e-6

# 水的注入速度
mesh.nodedata['Fw'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['Fw'][location1] = 3.5e-6


mesh.meshdata['water'] = {
    'viscosity': 1, # 1 cp = 1 mPa*s
    'compressibility': 1.0e-3, # MPa^{-1}
    }

mesh.meshdata['oil'] = {
    'viscosity': 2, # cp
    'compressibility': 2.0e-3, # MPa^{-1}
    }


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=isFractureCell)
mesh.find_node(axes, index=location0, showindex=True)
mesh.find_node(axes, index=location1, showindex=True)
plt.show()
