from fealpy.mesh import QuadrangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
import matplotlib.pyplot as plt

mesh = QuadrangleMesh.from_one_quadrangle()
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
print(node)
print(edge)
print(cell)
      
n=2
A = mesh.uniform_refine(n,returnim=True)
print(A[0],A[1])
def linear_function(coodrs):
    return 2*coodrs[:,0] + 2*coodrs[:,1]+2
excat_value = linear_function(mesh.node)
val = linear_function(node)
test_value = val
for i in range(n):
    test_value = A[i].dot(test_value)
print(excat_value)
print(test_value)

# def Prolongation_matrix(mesh,numnode,numedge,numcell):
#     '''
#     l:the nodes of new mesh
#     '''

#     l = numnode+numedge+numcell

#     data = bm.zeros(numnode+2*numedge+4*numcell,dtype=bm.float64)
#     indices = bm.zeros(numnode+2*numedge+4*numcell,dtype=bm.int32)
#     indptr = bm.zeros(l+1,dtype=bm.int32)

#     #赋值
#     data[:numnode] = 1
#     data[numnode:numnode+2*numedge] = 1/2
#     data[numnode+2*numedge:] = 1/4

#     indices[:numnode] = bm.arange(numnode) 
#     indices[numnode:numnode+2*numedge] = edge2node.flatten()
#     indices[numnode+2*numedge:] = cell2node.flatten()

#     indptr[:numnode+1] = bm.arange(numnode+1)
#     indptr[numnode+1:numnode+numedge+1]=bm.arange(numnode+2,numnode+2*numedge+1,step=2)
#     indptr[numnode+numedge+1:] = bm.arange(numnode+2*numedge+4,numnode+2*numedge+4*numcell+1,step=4)
#     A = CSRTensor(indptr,indices,data)
#     return A


# A = Prolongation_matrix(mesh,node_old,edge_old,cell_old)
# print(f"加密一次后的插值矩阵是:{A}")

# #带入线性函数验证
# def linear_function(coords):
#     return 5*coords[:, 0] + 4*coords[:, 1] + 3
# #计算函数在插值点数值
# value_check = linear_function(node2)
# #运用插值矩阵生成加密后的插值点函数值
# value_old = linear_function(node1)
# value_new = A.matmul(value_old) 
# print(f"真函数值:{value_check}")
# print(f"加密后的插值点函数值：{value_new}")
# are_close = bm.allclose(value_check, value_new)
# print(f"是否为插值矩阵（考虑精度）：{are_close}\n")

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# print(mesh.number_of_nodes())
# plt.show()
