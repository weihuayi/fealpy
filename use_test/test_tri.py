from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
import matplotlib.pyplot as plt

#生成三角形
node = bm.tensor([[0,0],[1,0],[0,1]],dtype=bm.float64)
cell=bm.tensor([[0,1,2]],dtype=bm.int32)
mesh = TriangleMesh(node, cell)
NN = mesh.number_of_nodes()

#定义插值矩阵
def Prolongation_matrix(mesh,numnode,numedge):
    l = numnode+numedge
    data = bm.zeros(numnode+2*numedge,dtype=bm.float64)
    indices = bm.zeros(numnode+2*numedge,dtype=bm.int32)
    indptr = bm.zeros(l+1,dtype=bm.int32)
    sparse_shape = bm.tensor([l, numnode])
    data[:numnode] = 1
    data[numnode:] = 0.5
    indices[:numnode] = bm.arange(numnode) 
    indices[numnode:] = edge.flatten()
    indptr[:numnode+1] = bm.arange(numnode+1)
    indptr[numnode+1:]=bm.arange(numnode+2,numnode+2*numedge+1,step=2)
    A = CSRTensor(indptr,indices,data,sparse_shape)
    return A

#网格加密
k = 1#设置加密次数
for i in range(k):
    edge = mesh.entity('edge')
    nn = mesh.number_of_nodes()
    ng = mesh.number_of_edges()
    mesh.uniform_refine()
    node_new = mesh.entity('node')
    A = Prolongation_matrix(mesh,numnode=nn,numedge=ng)
    #print(A)
print(f"加密1次后的插值矩阵：{A}")

#带入线性函数验证
def linear_function(coords):
    return coords[:, 0]**2 + 2*coords[:, 1] + 3
#计算函数在插值点数值
value_check = linear_function(node_new)
#运用插值矩阵生成加密后的插值点函数值
value = linear_function(node)
value_new = A.matmul(value) 
print(f"真函数值:{value_check}")
print(f"加密后的插值点函数值：{value_new}")
are_close = bm.allclose(value_check, value_new)
print(f"是否为插值矩阵（考虑精度）：{are_close}\n")