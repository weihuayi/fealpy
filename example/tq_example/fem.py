import taichi as ti


from fealpy.mesh import MeshFactory as MF

GD = 2

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 全局变量

node = ti.field(ti.float64, (NN, 2))
cell = ti.field(ti.int32, (NC, 3)) # 存储未知量
gphi = ti.field(ti.float64, (NC, 3, 2))

node.from_numpy(mesh.entity('node'))
cell.from_numpy(mesh.entity('cell'))

# 计算每个单元上的重心坐标的梯度
@ti.kernel
def grad_lambda():
    for c in range(NC):
        x0 = node[cell[c, 0], 0]
        y0 = node[cell[c, 0], 1]

        x1 = node[cell[c, 1], 0]
        y1 = node[cell[c, 1], 1]

        x2 = node[cell[c, 2], 0]
        y2 = node[cell[c, 2], 1]

        l = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) 
        gphi[c, 0, 0] = (y1 - y2)/l
        gphi[c, 0, 1] = (x2 - x1)/l 

        gphi[c, 1, 0] = (y2 - y0)/l
        gphi[c, 1, 1] = (x0 - x2)/l 

        gphi[c, 2, 0] = (y0 - y1)/l
        gphi[c, 2, 1] = (x1 - x0)/l 

grad_lambda()

gphi0 = mesh.grad_lambda()
gphi1 = gphi.to_numpy()

print(gphi0 - gphi1)
