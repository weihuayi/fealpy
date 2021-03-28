
import argparse
import pickle

from fealpy.writer import VTKMeshWriter

from TwoFluidsWithGeostressSimulator import TwoFluidsWithGeostressSimulator

from mumps import DMumpsContext
"""
python3 water_flooding_model_2d_test.py --mesh waterflooding_u32.pickle --T1 10 --DT 60 --step 1
"""

## 参数解析

parser = argparse.ArgumentParser(description=
        """
        这是一个两流体与地应力耦合的模拟程序。可计算

        * 水的饱和度 \n

        * 流体速度(m/s)\n

        * 压强(Pa)\n

        * 岩石位移 (m)\n 

        * 应力\n

        """)

parser.add_argument('--mesh', 
        default='waterflooding.pickle', type=str,
        help='需要导入的地质网格模型, 默认为 waterflooding.pickle, 如果 --reload 参数不是 None, 则该参数不起作用')

parser.add_argument('--T0', 
        default=0, type=float,
        help='模拟开始时间, 单位是天， 默认为 0 天，模拟程序内部会转换为秒')

parser.add_argument('--T1', 
        default=1, type=float, 
        help='模拟开始时间, 单位是天， 默认为 1 天，模拟程序内部会转换为秒')

parser.add_argument('--DT', 
        default=60, type=float,
        help='模拟时间步长, 单位是分种， 默认为 60 分种，模拟程序内部会转换为秒')

parser.add_argument('--step', 
        default=24, type=int,
        help='结果输出的步数间隔，默认为 24 步输出一次 vtu 文件')

parser.add_argument('--npicard', 
        default=20, type=int,
        help='Picard 迭代次数， 默认 20 次')

parser.add_argument('--output', 
        default='test', type=str,
        help='结果输出文件的主名称，默认为 test')

parser.add_argument('--save', 
        default='run.pickle', type=str,
        help='程序结束时，用于保存模拟器状态的文件名，用于续算')

parser.add_argument('--reload',
        default=[None, None], nargs=2,
        help='导入保存的运行环境，增加更多时间步, 如 --reload simulator.pickle 10，导入 simulator.pickle 文件，在原来的基础上多算 10 天')

parser.print_help()
args = parser.parse_args()

# 单位转换
args.T0 *= 3600*24 # 由天转换为秒
args.T1 *= 3600*24 # 由天转换为秒
args.DT *= 60 # 由分钟转换为 秒

# 打印当前所有参数
print(args) 

def water(s):
    return s**2

def oil(s):
    return (1-s)**2

if args.reload[0] is not None:
    n = int(float(args.reload[1])*3600*24/args.DT)
    with open(args.reload[0], 'rb') as f:
        simulator = pickle.load(f)
    simulator.add_time(n)

    #writer = VTKMeshWriter(simulation=simulator.run)
    #writer.run()

    writer = VTKMeshWriter()
    simulator.run(ctx=ctx, writer=writer)
else:

    ctx = DMumpsContext()
    ctx.set_silent()
    with open(args.mesh, 'rb') as f:
        mesh = pickle.load(f) # 导入地质网格模型

    mesh.fluid_relative_permeability_0 = water 
    mesh.fluid_relative_permeability_1 = oil 

    simulator = TwoFluidsWithGeostressSimulator(mesh, args)
    writer = VTKMeshWriter(simulation=simulator.run, args=(ctx, None))
    writer.run()
    ctx.destroy()

# 保存程序终止状态，用于后续计算测试
with open(args.save, 'wb') as f:
    pickle.dump(simulator, f, protocol=pickle.HIGHEST_PROTOCOL)

