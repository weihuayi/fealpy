import argparse

# 参数解析
parser = argparse.ArgumentParser(description="""
        Solve elastoplasticity problems using the finite element method.
        """)

parser.add_argument('--pde',
                    default=3, type=int,
                    help='Index of the elastoplasticity model, default is 3.')

parser.add_argument('--space_degree',
                    default=1, type=int,
                    help='Polynomial degree for the finite element space.')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='Show progress bar log.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Logging level. Default is INFO.')
# 解析参数                                                                                                                                                                                                                  
options = vars(parser.parse_args())

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

from fealpy.csm.fem import ThickWalledCylinderFEMModel
model = ThickWalledCylinderFEMModel(options)
# 网格可视化
from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh = model.mesh
mesh.add_plot(axes) # 画出网格背景

pde= model.pde
"""
node = mesh.entity('node')
bdnode = node[ pde.dirichlet_boundary(node) ]
innode = node[ pde.neumann_boundary(node) ]
import numpy as np
# 将 node 转为数组并绘制
coords = np.asarray(node[innode])
print(coords)
if coords.ndim == 2 and coords.shape[1] >= 2:
    # 绘制节点点
    axes.scatter(coords[:, 0], coords[:, 1], s=30, c='red', zorder=5)
    # 标注节点索引（可选，注释掉以隐藏标签）
    for i, p in enumerate(coords):
        axes.text(p[0], p[1], str(i),
                  fontsize=8, color='blue',
                  verticalalignment='bottom',
                  horizontalalignment='right')
    axes.set_aspect('equal', adjustable='box')
else:
    # 若为 1D 或 3D，降维到前两个分量绘制，或回退到 mesh.find_node
    try:
        axes.scatter(coords[:, 0], coords[:, 1], s=30, c='red', zorder=5)
    except Exception:
        mesh.find_node(axes, showindex=True)
plt.show()
"""

model.solve()


