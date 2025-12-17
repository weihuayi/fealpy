from fealpy.mesh import TetrahedronMesh

mesh = TetrahedronMesh.from_box_minus_cylinder()


# 网格可视化
from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes) # 画出网格背景
plt.show()