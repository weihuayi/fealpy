
import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory


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

for i in range(14):
    isFractureCell = is_fracture_cell(mesh)
    mesh.bisect(isFractureCell)

# 构建模型和数据参数

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
isFractureCell = is_fracture_cell(mesh)

# 渗透率
mesh.celldata['permeability'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['permeability'][isFractureCell] = 6 # 裂缝 1 d = 9.869 233e-13 m^2
mesh.celldata['permeability'][~isFractureCell] = 2 # 岩石 1 d = 9.869 233e-13 m^2

# 孔隙度
mesh.celldata['porosity'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['porosity'][ isFractureCell] = 0.3 # 裂缝
mesh.celldata['porosity'][~isFractureCell] = 0.3 # 岩石

# 拉梅第一常数, TODO:裂缝和岩石数值不一样
mesh.celldata['lambda'] = 1.0e+2 # MPa

# 拉梅第二常数, TODO:裂缝和岩石数值不一样
mesh.celldata['mu'] = 3.0e+2 # MPa

# Biot 系数，  TODO:裂缝和岩石数值不一样
mesh.celldata['biot'] = 1.0 

# MPa 固体体积模量, TODO: 裂缝和岩石数值不一样
mesh.celldata['solid grain stiffness'] = 2.0e+2 

# 初始压强
mesh.celldata['pressure'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['pressure'][:] = 3.0 # MPa

# 初始应力
mesh.celldata['stress'] = 2.0e+2 # MPa 初始应力 sigma_0, sigma_eff

# 水的饱和度
mesh.celldata['saturation'] = np.zeros(NC, dtype=np.float64)

# 水的注入速度
mesh.nodedata['injection'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['injection'][0] = 3.5e-6
mesh.nodedata['injection'][10] = 3.5e-6

# 油的开采速度
mesh.nodedata['production'] = np.zeros(NN, dtype=np.float64)
mesh.nodedata['production'][0] = 7.0e-6


mesh.meshdata['water'] = {
    'viscosity': 1, # 1 cp = 1 mPa*s
    'compressibility': 1.0e-3, # MPa^{-1}
    }

mesh.meshdata['oil'] = {
    'viscosity': 2, # cp
    'compressibility': 2.0e-3, # MPa^{-1}
    }


isFractureCell = is_fracture_cell(mesh)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#mesh.find_cell(axes, index=isFractureCell)
#mesh.find_node(axes, showindex=True)
plt.show()
