import matplotlib.pyplot as plt

from fealpy.backend import bm
bm.set_backend('pytorch')

from fealpy.ml.poisson_penn_model import PoissonPennModel
options = PoissonPennModel.get_options()

options = {"pde": 3}
model = PoissonPennModel(options)
pde = model.pde
print(pde.__doc__)

mesh = model.pde.init_mesh(80, 80)
net = model.net
node = mesh.entity('node')
val = model.shape_function(node)
fig = plt.figure()
axis = fig.add_subplot(projection='3d')
suf1 = axis.plot_trisurf(node[:,0], node[:,1], val,
                         cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.8)

fig.colorbar(suf1, ax=axis, shrink=0.5, label='Value')
plt.show()

# p = bm.tensor([0, 0.5, 0.7, 1], dtype=bm.float32)
# p =bm.tensor([[-1, 0.2], [1, 0.4], [0.5, 0.6], [0.5, 0.5],[0.7, -1], [0,0], [0, 1]], dtype=bm.float32)
# a = model.shape_function(p)
# print(a)

# print(net)
# print(model.optimizer)
# print(model.activation)
# print(model.steplr)
# fig = plt.figure()
# axis = fig.add_subplot()
# mesh.add_plot(axis)
# plt.show()
# print(dir(model))


