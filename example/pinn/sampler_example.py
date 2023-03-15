
from fealpy.pinn.sampler import (
    TriangleMeshSampler,
    ISampler,
    JoinedSampler,
    HybridSampler
)
from fealpy.mesh import MeshFactory as Mf


print('1. 使用 TriangleMeshSampler 从三角形网格中取得样本（如图）')

mesh = Mf.boxmesh2d([0, 1, 0, 1], nx=2, ny=2)

tms = TriangleMeshSampler(3, mesh=mesh)
samples = tms.run()
print(tms.run())


### Draw the result

from matplotlib import pyplot as plt

fig, axes = plt.subplots()
mesh.add_plot(axes)
axes.scatter(samples[:, 0], samples[:, 1])
plt.show()


print('2. 使用 ISampler 取得各轴独立的样本')

s2 = ISampler(10, [[0, 1], [2, 3], [4, 5]])
print(s2.run())


print('3. JoinedSampler 会拼接各子采样器的样本，但是子采样器必须产生相同的特征数')

s3 = ISampler(5, [[0, 1]]) & ISampler(3, [[4, 6]])
print(s3.run())


print('4. HybridSampler 会拼接各子采样器的特征，但是子采样器必须产生相同的样本数')

s4 = ISampler(5, [[0, 1], [6, 7]]) | ISampler(5, [[-1, -2]])
print(s4.run())
