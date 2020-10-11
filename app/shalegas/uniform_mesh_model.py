
import pickle

import numpy as np
from scipy.spatial import KDTree 

import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory


"""

Notes
-----
均匀二维网格模型， 把模型保存为 pickle 文件 
"""


box = [0, 10, 0, 10] # m 
mesh = MeshFactory.boxmesh2d(box, nx=32, ny=32, meshtype='tri')
mesh.box = box


# 构建模型和数据参数
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 构建 KDTree，用于定义注入和生产井的网格点位置
node = mesh.entity('node')
tree = KDTree(node)

p0 = np.array([(10, 10)], dtype=np.float64) # 生产井位置
p1 = np.array([(0, 0)], dtype=np.float64) # 注水井位置 

_, location0 = tree.query(p0)
_, location1 = tree.query(p1)

print('生产井的位置:', node[location0])
print('注水井的位置:', node[location1])


# 裂缝标签
mesh.celldata['fracture'] = np.zeros(NC, dtype=np.int_)

# 渗透率，裂缝和岩石的渗透率不同
mesh.celldata['permeability'] = np.zeros(NC, dtype=np.float64) # 1 d = 9.869 233e-13 m^2
mesh.celldata['permeability'][:] = 2 

# 孔隙度
mesh.celldata['porosity'] = np.zeros(NC, dtype=np.float64) # 百分比
mesh.celldata['porosity'][:] = 0.3 # 裂缝

# 拉梅第一常数
mesh.celldata['lambda'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['lambda'][:] = 1.0e+2 # 裂缝 

# 拉梅第二常数
mesh.celldata['mu'] =  np.zeros(NC, dtype=np.float64) # MPa
mesh.celldata['mu'][:] = 3.0e+2 # 裂缝 

# Biot 系数, TODO: 岩石和裂缝不同
mesh.celldata['biot'] =  np.zeros(NC, dtype=np.float64) # 
mesh.celldata['biot'][:] = 1.0 # 裂缝 

# MPa 固体体积模量 K = lambda + 2*mu/3
mesh.celldata['K'] =  mesh.celldata['lambda'] + 2*mesh.celldata['mu']/3 

# 初始压强
mesh.celldata['pressure'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['pressure'][:] = 3.0 # MPa

# 初始应力
mesh.celldata['stress'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['stress'][:] = 2.0e+2 # MPa 初始应力 sigma_0, sigma_eff

# 初始水的饱和度 
mesh.celldata['fluid_0'] = np.zeros(NC, dtype=np.float64)
# 初始气或油的饱和度 
mesh.celldata['fluid_1'] = 1 - mesh.celldata['fluid_0'] 

# 流体 0 的源汇项
mesh.nodedata['source_0'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['source_0'][location1] = 3.51e-6 # 正号表示注入

# 流体 1 的源汇项
mesh.nodedata['source_1'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['source_1'][location0] = -3.5e-6 # 负值表示产出



mesh.meshdata['fluid_0'] = {
    'name': 'water',
    'viscosity': 1, # 1 cp = 1 mPa*s
    'compressibility': 1.0e-3, # MPa^{-1}
    }

mesh.meshdata['fluid_1'] = {
    'name': 'oil', 
    'viscosity': 2, # cp
    'compressibility': 2.0e-3, # MPa^{-1}
    }

with open('waterflooding_u32.pickle', 'wb') as f:
    pickle.dump(mesh, f, protocol=pickle.HIGHEST_PROTOCOL)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, index=location0, showindex=True)
mesh.find_node(axes, index=location1, showindex=True)
plt.show()
