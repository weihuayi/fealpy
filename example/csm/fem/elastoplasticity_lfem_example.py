import argparse

# 参数解析
parser = argparse.ArgumentParser(description="""
        Solve elastoplasticity problems using the finite element method.
        """)

parser.add_argument('--pde',
                    default=1, type=int,
                    help='Index of the elastoplasticity model, default is 1.')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='Show progress bar log.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Logging level. Default is INFO.')
# 解析参数
options = vars(parser.parse_args())

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.csm.fem import ElastoplasticityFEMModel
model = ElastoplasticityFEMModel(options)
mesh = model.mesh
'''
# 网格可视化
from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes) # 画出网格背景
mesh.find_cell(axes, showindex=True) # 找到单元重心
plt.show()
'''