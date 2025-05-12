from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.fdm.laplace_operator import LaplaceOperator
from fealpy.mesh import UniformMesh
from fealpy.fdm import DirichletBC
from fealpy.solver import spsolve
import matplotlib.pyplot as plt
from fealpy.model.poisson import get_example, example

example = Example()
example.show_examples()
pde = example.get_example('coscos')

# pde = get_example('sinsin', flag=True)

print(pde.domain())
exit()


domain = pde.domain()
extent = [0, 10, 0, 10]
mesh = UniformMesh(domain,extent)
maxit = 5
em = bm.zeros((3, maxit), dtype=bm.float64)
for i in range(maxit):
    A = LaplaceOperator(mesh).assembly()
    f = mesh.interpolate(pde.source)
    dbc = DirichletBC(mesh=mesh, gd=pde.dirichlet)
    A, f = dbc.apply(A, f)
    uh = spsolve(A,f,solver='scipy')
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    if i < maxit-1:
        mesh.uniform_refine()
print("em:\n", em)
print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])

fig = plt.figure(1)
axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh.reshape(mesh.nx+1, mesh.ny+1))
plt.title(f"Iteration {i+1}")
fig, axes = plt.subplots()
error_names = ['max', 'L2', 'l2']  # 定义误差名称
markers = ['o-', 's--', '^:']      # 定义不同线条样式
for i in range(3):
    axes.plot(em[i, :], markers[i], label=error_names[i])
axes.set_xlabel('Refinement Level')
axes.set_ylabel('Error')
axes.set_xticks(bm.arange(maxit))  # 设置 x 轴刻度
axes.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.title('Error Convergence')
plt.show()