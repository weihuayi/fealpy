import argparse
from fealpy.backend import backend_manager as bm
from fealpy.csm.fem import ElastoplasticityFEMModel

# 参数解析
parser = argparse.ArgumentParser(description="""
        用有限元方法计算弹塑性问题的位移
        """)

parser.add_argument('--pde',
                    default='1', type=str,
                    help='选择预设的弹塑性问题示例，默认为"1"')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='是否显示进度条日志.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                        help='日志级别, 默认为 INFO.')
# 解析参数
options = vars(parser.parse_args())

bm.set_backend('numpy')

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