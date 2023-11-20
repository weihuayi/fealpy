import argparse
from paraview_plotter import ParaViewPlotter

# 设置命令行参数
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', type=str, help='The path to the VTK file to visualize.')
parser.add_argument('--background_color', 
        type=tuple, default=(1.0, 1.0, 1.0), help='The background color of the visualization.')
parser.add_argument('--show_type', type=str, default='Surface With Edges')

# 解析命令行参数
args = parser.parse_args()

# 创建 ParaViewPlotter 对象
plotter = ParaViewPlotter(args)
# 执行画图
plotter.plot()

