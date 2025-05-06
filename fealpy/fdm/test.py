from fealpy.fdm import LaplaceOperator
from fealpy.mesh import UniformMesh

domain = [0.0, 1.0]
extent = [0, 10]
mesh = UniformMesh(domain, extent)

L0 = LaplaceOperator(mesh)

A = L0.assembly()

# method 和 call 方法只需要使用一种
L01 = LaplaceOperator(mesh, method='fast_assembly')
A1 =L01()
print("------------")