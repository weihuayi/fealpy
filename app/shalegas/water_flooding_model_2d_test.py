
import argparse
import pickle

from WaterFloodingWithFractureModel2d import WaterFloodingWithFractureModel2d 
from TwoPhaseFlowWithGeostressSimulator import TwoPhaseFlowWithGeostressSimulator


## 参数解析

parser = argparse.ArgumentParser(description=
        """
        这是一个水驱油或气的模拟程序。可计算

        * 水的饱和度 \n

        * 流体速度(m/s)\n

        * 压强(Pa)\n

        * 岩石位移 (m)\n 

        * 应力\n

        """)

parser.add_argument('--T0', 
        default=0, type=float,
        help='模拟开始时间, 单位是天， 默认为 0 天，模拟程序内部会转换为秒')

parser.add_argument('--T1', 
        default=1, type=float, 
        help='模拟开始时间, 单位是天， 默认为 1 天，模拟程序内部会转换为秒')

parser.add_argument('--DT', 
        default=1, type=float,
        help='模拟时间步长, 单位是分种， 默认为 1 分种，模拟程序内部会转换为秒')

parser.add_argument('--NS', 
        default=14, type=int,
        help='裂缝附近加密次数，默认 14 次')

parser.add_argument('--step', 
        default=60, type=int,
        help='结果输出的步数间隔，默认为 60 步输出一次 vtu 文件')

parser.add_argument('--output', 
        default='test', type=str,
        help='结果输出文件的主名称，默认为 test')

parser.add_argument('--save', 
        default='run.pickle', type=str,
        help='程序结束时，用于保存模拟器状态的文件名，用于续算')

parser.add_argument('--reload',
        default=[None, None], nargs=2,
        help='导入保存的运行环境，增加更多时间步, 如 --reload fname.pickle 10，导入 fname.pickle 文件，在原来的基础上多算 10 天')

parser.add_argument('--showmesh',
        default=False, type=bool,
        help='是否展示计算网格，默认为 False')

parser.print_help()
args = parser.parse_args()
# 打印当前所有参数
print(args) 

# 单位转换
args.T0 *= 3600*24 # 由天转换为 秒
args.T1 *= 3600*24 # 由天转换为秒
args.DT *= 60 # 由分钟转换为 秒


if args.reload[0] is not None:
    n = int(float(args.reload[1])*3600*24/args.DT)
    with open(args.reload[0], 'rb') as f:
        solver = pickle.load(f)
    solver.add_time(n)
else:
    model = WaterFloodingWithFractureModel2d()
    solver = TwoPhaseFlowWithGeostressSimulator(model, args)

if args.showmesh is True:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.gca()
    mesh = solver.mesh
    mesh.add_plot(axes)
    bdNodeIdx = mesh.ds.boundary_node_index()
    mesh.find_node(axes, index=bdNodeIdx, showindex=True)
    plt.show()
else:
    if args.reload[0] is not None:
        solver.solve(step=args.step, reset=False)
    else:
        solver.solve()

    # 保存程序终止状态，用于后续计算测试
    with open(args.save, 'wb') as f:
        pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)

