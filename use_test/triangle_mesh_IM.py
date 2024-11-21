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
            data[indptr[i]-1] = 2/(indptr[i+1]-indptr[i]+1)
            data[indptr[i]:indptr[i+1]-1] = 1/(indptr[i+1]-indptr[i]+1)

    R = CSRTensor(indptr,indices,data)
    return R

R = Resriction_matrix(mesh,node_H,node_h)

def linear_function(coodrs):
    return coodrs[:,0] + 2*coodrs[:,1]+3
value_check = linear_function(mesh.node)
value_new = R.matmul(value_check)
value_real = linear_function(node_new)

print(f"限制矩阵：{R}\n")
print(f"精确值:{value_real},检验限制矩阵的值：{value_new}")
