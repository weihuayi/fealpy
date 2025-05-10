from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.model.poisson import get_example

from fealpy.mesh import UniformMesh
from fealpy.fdm import LaplaceOperator
from fealpy.fdm import DirichletBC
from fealpy.solver import spsolve
import matplotlib.pyplot as plt


pde = get_example('sin') # 获取 PDE 模型
domain = pde.domain()

extent = [0, 10]
mesh = UniformMesh(domain=domain, extent=extent)

maxit = 5   # 网格加密次数
em = bm.zeros((3, maxit), dtype=bm.float64)

for i in range(maxit):
    A = LaplaceOperator(mesh=mesh).assembly()
    node = mesh.entity("node")
    F = pde.source(p=node)

    bc = DirichletBC(mesh=mesh, gd=pde.dirichlet)
    A, F = bc.apply(A, F)

    NN = mesh.number_of_nodes()
    print(i, NN)

    uh = spsolve(A, F, solver='scipy')

    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh, errortype='all')
    
    if i < maxit-1:
        mesh.uniform_refine() 

em_ratio = em[:, 0:-1] / em[:, 1:]
print("误差: ", em, "误差比: ",em_ratio,sep='\n')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  

# 绘制误差曲线（左图）
error_names = ['max', 'L2', 'H1']
markers = ['o-', 's--', '^:']

for i in range(3):
    ax1.plot(em[i, :], markers[i], label=error_names[i], linewidth=4) 
ax1.set_xlabel('Refinement Level', fontsize=24) 
ax1.legend(fontsize=20)  
ax1.grid(True)
ax1.set_title(' Error ', fontsize=28)  

# 绘制误差比的曲线（右图）
for i in range(3):
    ax2.plot(em_ratio[i, :], markers[i], label=f'{error_names[i]} ratio', linewidth=4)
ax2.axhline(y=4, color='r', linestyle='-', label='y=4 (expected ratio)', linewidth=4)
ax2.set_xlabel('Refinement Level', fontsize=24)
ax2.legend(fontsize=20)  
ax2.grid(True)
ax2.set_title('Error Ratio', fontsize=28)

plt.show()