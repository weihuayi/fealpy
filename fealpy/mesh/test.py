from fealpy.mesh import HexahedronMesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建两个六面体网格
mesh1 = HexahedronMesh.from_one_hexahedron()
mesh2 = HexahedronMesh.from_one_hexahedron()
mesh2.uniform_refine()

# 创建一个图形和两个子图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# 绘制 mesh1
mesh1.add_plot(ax1)
mesh1.find_node(ax1, showindex=True)
mesh1.find_edge(ax1, showindex=True)
ax1.set_title("Mesh 1")

# 绘制 mesh2
mesh2.add_plot(ax2)
mesh2.find_node(ax2, showindex=True)
mesh2.find_edge(ax2, showindex=True)
ax2.set_title("Mesh 2: Refine")

# 设置三维坐标轴的标签
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# 设置三维坐标轴的比例
ax1.set_box_aspect([1, 1, 1])
ax2.set_box_aspect([1, 1, 1])

# 设置视角
ax1.view_init(elev=30, azim=45)
ax2.view_init(elev=30, azim=45)

# 显示图形
plt.tight_layout()
plt.show()