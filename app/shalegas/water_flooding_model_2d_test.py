import sys
import argparse

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

parser.add_argument('--dir', 
        default='./', type=str,
        help='结果输出的目录，默认为当前目录')

parser.add_argument('--step', 
        default=60, type=str,
        help='结果输出的步数间隔，默认为 60')

parser.add_argument('--fname', 
        default='test', type=str,
        help='结果输出文件的主名称，默认为 test')

parser.add_argument('--save', 
        default=True, type=bool,
        help='程序结束时，是否保存程序的运行环境，用于续算， 默认为 True')

parser.add_argument('--reload',
        default=[None, None], nargs=2,
        help='导入保存的运行环境，增加更多时间步')

args = parser.parse_args()

# 单位转换
args.T0 *= 3600*24
args.T1 *= 3600*24
args.DT *= 60
print(args)

model = WaterFloodingWithFractureModel2d()
solver = TwoPhaseFlowWithGeostressSimulator(model)
#solver.solve()
