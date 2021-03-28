import pickle

import numpy as np
from scipy.spatial import KDTree 

import matplotlib.pyplot as plt
from matplotlib import collections  as mc

from fealpy.mesh import MeshFactory


"""

Notes
-----
用于制作带裂缝的两相流网格模型, 并把模型保存为 pickle 文件 
"""


box = [0, 10, 0, 10] # m 
mesh = MeshFactory.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

mesh.box = box
for i in range(10):
    mesh.bisect()

point = np.array([
    (0.5, 5.0), #0
    (9.5, 5.0), #1
    (1.0, 2.0), #2
    (1.0, 8.0), #3 
    (3.0, 3.0), #4
    (3.0, 7.0), #5
    (5.0, 2.0), #6
    (5.0, 8.0), #7
    (7.0, 1.0), #8
    (7.0, 9.0), #9
    (9.0, 4.0), #10
    (9.0, 6.0), #11
    ], dtype=np.float64)

segment = np.array([
    (0, 1), 
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    ], dtype=np.int_)

for i in range(8):
    isFractureCell = mesh.is_crossed_cell(point, segment) 
    mesh.bisect(isFractureCell)

# 构建模型和数据参数
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 构建 KDTree，用于定义注入和生产井的网格点位置
node = mesh.entity('node')
tree = KDTree(node)

p0 = np.array([(0, 0), (0, 10)], dtype=np.float64) # 注水井位置 
p1 = np.array([(9.5, 5)], dtype=np.float64) # 生产井位置

_, location0 = tree.query(p0)
print('注水井的位置:', node[location0])

_, location1 = tree.query(p1)
print('生产井的位置:', node[location1])


# 裂缝标签
isFractureCell = mesh.is_crossed_cell(point, segment)

# vtk 不支持 bool 类型
mesh.celldata['fracture'] = np.asarray(isFractureCell, dtype=np.int_)

# 渗透率
mesh.celldata['permeability'] = np.zeros(NC, dtype=np.float64) # 1 d = 9.869 233e-13 m^2
mesh.celldata['permeability'][ isFractureCell] = 0.06 # 裂缝 
mesh.celldata['permeability'][~isFractureCell] = 0.02 # 岩石 

# 孔隙度
mesh.celldata['porosity'] = np.zeros(NC, dtype=np.float64) # 百分比
mesh.celldata['porosity'][ isFractureCell] = 0.05 # 裂缝
mesh.celldata['porosity'][~isFractureCell] = 0.1 # 岩石

# 拉梅第一常数
mesh.celldata['lambda'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['lambda'][ isFractureCell] = 0.5e+2 # 裂缝 
mesh.celldata['lambda'][~isFractureCell] = 1.0e+2 # 岩石 

# 拉梅第二常数
mesh.celldata['mu'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['mu'][ isFractureCell] = 1.5e+2 # 裂缝 
mesh.celldata['mu'][~isFractureCell] = 3.0e+2 # 岩石 

# Biot 系数
mesh.celldata['biot'] =  np.zeros(NC, dtype=np.float64) # 
mesh.celldata['biot'][ isFractureCell] = 1.0 # 裂缝 
mesh.celldata['biot'][~isFractureCell] = 1.0 # 岩石 

# MPa 固体体积模量 K = lambda + 2*mu/3
mesh.celldata['K'] =  mesh.celldata['lambda'] + 2*mesh.celldata['mu']/3

# 初始压强
mesh.celldata['pressure'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['pressure'][:] = 3.0 # MPa

# 初始应力
mesh.celldata['stress'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['stress'][:] = 1.0e+2 # MPa 初始应力 sigma_0 + sigma_eff

# 初始 0 号流体的饱和度 
mesh.celldata['fluid_0'] = np.zeros(NC, dtype=np.float64)
# 初始 1 号流体的饱和度 
mesh.celldata['fluid_1'] = 1 - mesh.celldata['fluid_0'] 

# 0 号流体的注入速度
mesh.nodedata['source_0'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['source_0'][location0] = 3.51e-6

# 1 号流体的开采速度
mesh.nodedata['source_1'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['source_1'][location1] = -7.0e-6


# 0 号流体的性质
mesh.meshdata['fluid_0'] = {
    'name': 'water',
    'viscosity': 1, # 1 cp = 1 mPa*s
    'compressibility': 1.0e-3, # MPa^{-1}
    }

# 1 号流体的性质
mesh.meshdata['fluid_1'] = {
    'name': 'oil', 
    'viscosity': 2, # cp
    'compressibility': 2.0e-3, # MPa^{-1}
    }

with open('fracture_model_1.pickle', 'wb') as f:
    pickle.dump(mesh, f, protocol=pickle.HIGHEST_PROTOCOL)

fig = plt.figure()
lc = mc.LineCollection(point[segment], linewidths=2, color='r')
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=isFractureCell)
mesh.find_node(axes, index=location0, showindex=True)
mesh.find_node(axes, index=location1, showindex=True)
axes.add_collection(lc)
plt.show()
