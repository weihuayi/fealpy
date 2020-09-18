
import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory


def is_crossed_cell(mesh):

    NC = mesh.number_of_cells()

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    isCrossedCell = np.zeros(NC, dtype=np.bool_)

    def f(x):
        a = x[0]
        b = x[1]
        d = x[2]
        flag0 = (node[cell[:, 0], d[0]] >= b[0]) & (node[cell[:, 0], d[0]] <= b[1]) & (node[cell[:, 0], d[1]] == a)
        flag1 = (node[cell[:, 1], d[0]] >= b[0]) & (node[cell[:, 1], d[0]] <= b[1]) & (node[cell[:, 1], d[1]] == a)
        flag2 = (node[cell[:, 2], d[0]] >= b[0]) & (node[cell[:, 2], d[0]] <= b[1]) & (node[cell[:, 2], d[1]] == a)
        isCrossedCell[flag0 | flag1 | flag2] = True

    # fracture
    a = [5, 1, 3, 5, 7, 9]
    b = [(0.5, 9.5), (2, 8), (3, 7), (2, 8), (1, 9), (4, 6)]
    d = [(0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    list(map(f, zip(a, b, d)))

    return isCrossedCell

box = [0, 10, 0, 10] # m 
mesh = MeshFactory.boxmesh2d(box, nx=10, ny=10, meshtype='tri')

for i in range(14):
    isCrossedCell = is_crossed_cell(mesh)
    mesh.bisect(isCrossedCell)

# 构建模型和数据参数

NC = mesh.number_of_cells()
isCrossedCell = is_crossed_cell(mesh)

mesh.celldata['permeability'] = np.zeros(NC, dtype=np.float64)
mesh.celldata['permeability'][isCrossedCell] = 6 # 1 d = 9.869 233e-13 m^2
mesh.celldata['permeability'][isCrossedCell] = 2 # 1 d = 9.869 233e-13 m^2


mesh.meshdata['rock'] = {
    'permeability': 2, # 1 d = 9.869 233e-13 m^2 
    'porosity': 0.3, # None
    'lame':(1.0e+2, 3.0e+2), # lambda and mu 拉梅常数, MPa
    'biot': 1.0,
    'initial pressure': 3, # MPa
    'initial stress': 2.0e+2, # MPa 初始应力 sigma_0 , sigma_eff
    'solid grain stiffness': 2.0e+2, # MPa 固体体积模量
    }

mesh.meshdata['fracture'] = {
    'permeability': 6, # 1 d = 9.869 233e-13 m^2 
    'porosity': 0.1, # None
    'lame':(1.0e+1, 3.0e+1), # lambda and mu 拉梅常数, MPa
    'solid grain stiffness': 1.0e+2, # MPa 固体体积模量
    }

mesh.meshdata['water'] = {
    'viscosity': 1, # 1 cp = 1 mPa*s
    'compressibility': 1.0e-3, # MPa^{-1}
    'initial saturation': 0.0, 
    'injection rate': 3.51e-6 # s^{-1}, 每秒注入多少水
    }


isCrossedCell = is_crossed_cell(mesh)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=isCrossedCell)
plt.show()
