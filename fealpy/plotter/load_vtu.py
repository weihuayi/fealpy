import argparse
from .ParaViewPlotter import ParaViewPlotter

# 设置命令行参数
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', help='The path to the VTK file to visualize.')
parser.add_argument('--background_color', default='white', help='The background color of the visualization.')
parser.add_argument('--show_edges', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--save_image_path', help='Path to save the rendered image.')

# 解析命令行参数
args = parser.parse_args()

# 创建 ParaViewPlotter 对象
plotter = ParaViewPlotter(
    filename=args.filename,
    background_color=args.background_color,
    show_edges=args.show_edges,
    save_image_path=args.save_image_path
)

# 执行画图
plotter.plot()

