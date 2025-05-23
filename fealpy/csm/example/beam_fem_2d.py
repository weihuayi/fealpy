import argparse
from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from fealpy.mesh import EdgeMesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 参数解析
parser = argparse.ArgumentParser(description="""
        用线性有限元方法计算梁结构的位移
        """)

parser.add_argument('--modulus',
                    default=29e6, type=float,
                    help='梁的杨氏模量, 默认为 29e6 ton/mm^2.')

parser.add_argument('--high',
                    default=17.7, type=float,
                    help='梁的高度, 默认为 17.7 ton/mm^2.')

parser.add_argument('--area',
                    default=10.3, type=float,
                    help='梁的宽度, 默认为 10.3 ton/mm^2.')

parser.add_argument('--inertia',
                    default=510, type=float,
                    help='梁的惯性矩, 默认为 510 ton/mm^2.')

args = parser.parse_args()

E = args.modulus
A = args.area
t = args.high
I = args.inertia

# 构造网格
node = bm.array([
    [0], [10]], dtype=bm.float64)
cell = bm.array([
    [0, 1]] , dtype=bm.int16)
mesh = EdgeMesh(node, cell)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()



# 组装刚度矩阵
l = mesh.cell_length().reshape(-1, 1)


l=l[0,0]
print(l)
K = bm.array([
            [12, 6*l, -12, 6*l],
            [6*l, 4*l**2, -6*l, 2*l**2],
            [-12, -6*l, 12, -6*l],
            [6*l, 2*l**2, -6*l, 4*l**2]], dtype=bm.float64)



# 右端项组装
f = 1000 # 向下施加均匀荷载
F = bm.array([ -1/2*l*f, -1/12*l*l*f, -1/2*l*f, 1/12*l*l*f], dtype=bm.float64)

F *=l**3
F /=E*I*1/12**2
#边界条件处理
D_0=bm.array([[0,0,0,0],
        [0,0,0,0],
        [0,0,1,0],
        [0, 0, 0, 1]])
D_1=bm.array([[1,0,0,0],
        [0,1,0,0],
        [0,0,0,0],
        [0, 0, 0, 0]])
K=D_0@K@D_0+D_1
F=D_0@F
#TODO: 这里的边界条件处理需要改进,写FEALPy统一的接口。

# 将矩阵转换为CSR格式
K= csr_matrix(K)

# 求解
uh = bm.zeros((NN, GD+1), dtype=bm.float64)
uh.T.flat = spsolve(K, F)

print(uh)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()


