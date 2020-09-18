
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
    b = [(1, 9), (2, 8), (3, 7), (2, 8), (1, 9), (4, 6)]
    d = [(0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    list(map(f, zip(a, b, d)))

    return isCrossedCell

box = [0, 10, 0, 10] # m 
mesh = MeshFactory.boxmesh2d(box, nx=10, ny=10, meshtype='tri')

for i in range(10):
    isCrossedCell = is_crossed_cell(mesh)
    mesh.bisect(isCrossedCell)


isCrossedCell = is_crossed_cell(mesh)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_cell(axes, index=isCrossedCell)
plt.show()
