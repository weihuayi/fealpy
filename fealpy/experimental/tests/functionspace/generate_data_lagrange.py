#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: generate_data.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 05 Aug 2024 03:22:05 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
from fealpy.decorator import barycentric, cartesian

def test_fun(point): 
    x = point[..., 0]
    y = point[..., 1]
    result = np.sin(x) * np.sin(y)
    return result

mesh = TriangleMesh.from_box([0,1,0,1], 1, 1)
space = LagrangeFESpace(mesh, 2)
uh = space.interpolate(test_fun)
bcs = np.array([[0, 0, 1], [0, 1, 0], [1/3, 1/3, 1/3]], dtype=np.float64)

#print(uh)
#print(space.value(uh, bcs).swapaxes(1,0))
print(space.grad_basis(bcs).swapaxes(0,1))
#print(space.grad_value(uh, bcs).shape)


'''
print(space.number_of_local_dofs())
print(space.number_of_global_dofs())
print(space.interpolation_points())
print(space.cell_to_dof())
print(space.face_to_dof())
print(space.edge_to_dof())
'''

'''
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=mesh.node,showindex=True,color='r')
mesh.find_edge(axes,showindex=True)
plt.show()
'''

"""
import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh    
from fealpy.functionspace import LagrangeFESpace

### 以下代码是一个示例，用于说明如何测试LagrangeFESpace类的boundary_interpolate方法

# 假设你已经定义了PDE类和LagrangeFESpace类，以及TriangleMesh类

class PDE:
    def __init__(self):
        pass

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(np.pi*x) * np.sin(np.pi*y)
    

# 创建PDE实例
pde = PDE()

# 创建网格和有限元空间
mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)
space = LagrangeFESpace(mesh, 2)

# 初始化uh，大小与自由度数量一致
uh = np.zeros(space.number_of_global_dofs())

# 设置边界条件
b = space.boundary_interpolate(gD=pde.solution, uh=uh)

# 输出结果
print("b:", b)
print("uh:", uh)
"""