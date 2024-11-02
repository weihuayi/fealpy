
from fealpy.ml.sampler import (
    MeshSampler,
    ISampler
)
from fealpy.mesh import TriangleMesh


print('1. 使用 MeshSampler 从网格中取得样本（如图）')

mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=2, ny=2)

tms = MeshSampler(mesh, 'cell')
samples = tms.run(3)
print(samples)


### Draw the result

from matplotlib import pyplot as plt

fig, axes = plt.subplots()
mesh.add_plot(axes)
axes.scatter(samples[:, 0], samples[:, 1])
plt.show()


print('2. 使用 ISampler 取得各轴独立的样本')

s2 = ISampler([[0, 1], [2, 3], [4, 5]])
print(s2.run(10))
