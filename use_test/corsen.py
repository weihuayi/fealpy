from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.sparse import CSRTensor
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

#生成三角形
node = bm.tensor([[0,0],[1,0],[0,1]],dtype=bm.float64)
cell=bm.tensor([[0,1,2]],dtype=bm.int32)
mesh = TriangleMesh(node, cell)
mesh.uniform_refine()

#确定粗网格节点
cell_h = mesh.number_of_cells()
cell_H = 0.25*cell_h
n_1 = symbols('n_1')
equation = Eq(n_1**2, cell_H)
solutions = solve(equation, n_1)
n_1 = bm.int32(solutions[1])+1
node_H = int((n_1**2+n_1)/2)
node_h = mesh.number_of_nodes()
node_new = mesh.entity('node')[:node_H]

edge2node = mesh.edge_to_node()

def Resriction_matrix(mesh,row1,row2):
    valence = bm.zeros(node_h, dtype=mesh.itype, device=mesh.device)
    bm.add_at(valence, edge2node, 1)
    row3 = node_H+bm.sum(valence[0:node_H])
    
    #indptr
    indptr = bm.zeros(row1+1,dtype=bm.int32)
    indptr[0]= 1
    for i in range(1,row1+1):
        indptr[i] = indptr[i-1]+1+valence[i-1]
    
    #indices
    indices = bm.zeros(row3,dtype=bm.int32)
    for k in range(row1):
        t_list = []
        for element in edge2node:
            if k in element:
                other_number = [x for x in element if x != k]
                t_list.append(other_number)
        t_list.sort()
        t = bm.array(t_list)
        indices[indptr[k]:indptr[k+1]-1] = t.flatten()
        indices[indptr[k]-1] = k
   
    data = bm.zeros(row3,dtype=bm.float64)
    for i in range(row1):
            print(indptr[i+1]-indptr[i])
            data[indptr[i]-1] = 2/(indptr[i+1]-indptr[i]+1)
            data[indptr[i]:indptr[i+1]-1] = 1/(indptr[i+1]-indptr[i]+1)

    R = CSRTensor(indptr,indices,data)
    return R

R = Resriction_matrix(mesh,node_H,node_h)

# # 求解方程
# sol = sp.solve([eq], [0])
# edge = mesh.entity('edge')
# edge2node = mesh.edge_to_node()
# NE = mesh.number_of_edges()

# for i in range(NE):
#     if (node[edge[i],:][0][0]!=node[edge[i],:][1][0]) and (node[edge[i],:][0][1]!=node[edge[i],:][1][1]):
#         line[i]=(node[edge[i],:][0][0]-node[edge[i],:][1][0])/(node[edge[i],:][0][1]-node[edge[i],:][1][1])
# isGEdge = (line!=0)
# print(edge[isGEdge])

# cell2edge = mesh.cell_to_edge()
# GEdge= bm.zeros(NE, dtype=mesh.itype, device=mesh.device)
# bm.add_at(GEdge, cell2edge, 1)
# print(GEdge)
# isGEdge = (GEdge==2)
# print(edge[isGEdge])


# GNode = bm.zeros(NE, dtype=mesh.itype, device=mesh.device)
# bm.add_at(GNode, cell2edge, 1)
# print(GNode)
# isGNode = (GNode != 0)
# print(isGNode)
# print(node[isGNode])

# print(mesh.cell_to_node())
# valence = bm.zeros(NN, dtype=mesh.itype, device=mesh.device)
# bm.add_at(valence, cell, 1)
# isBNode = (valence==1)
# BNode = node[isBNode]
# print(BNode)

print(f"限制矩阵为:{R}")
print(f"粗化新坐标点:{node_new}")
def linear_function(coodrs):
    return coodrs[:,0]*coodrs[:,0] + 2*coodrs[:,1]+3
value_check = linear_function(mesh.node)
value_new = R.matmul(value_check)
print(f"限制矩阵下的插值结果:{value_new}")
print(f"精确值:{linear_function(node_new)}")


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
plt.show()

# try_1 = bm.tensor([3.875,4.5,2.125],dtype = bm.float64)
# try_2 = CSRTensor(crow=bm.tensor([0 ,1 ,2 ,3 ,5 ,7 ,9]), col=bm.tensor([0 ,1 ,2 ,0 ,1 ,2 ,0 ,1 ,2]), 
#                   values=bm.tensor([1.  ,1.  ,1.  ,0.5 ,0.5 ,0.5 ,0.5 ,0.5 ,0.5]))
# print(f"check:{try_2.matmul(try_1)}")