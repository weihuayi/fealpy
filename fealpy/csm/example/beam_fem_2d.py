import argparse
from fealpy.backend import backend_manager as bm
from fealpy.csm.fem import BeamFEMModel

# 参数解析
parser = argparse.ArgumentParser(description="""
        用线性有限元方法计算梁结构的位移
        """)

parser.add_argument('--modulus',
                    default=200e9, type=float,
                    help='梁的杨氏模量, 默认为 200e9 Pa.')

parser.add_argument('--inertia',
                    default=118.6e-6, type=float,
                    help='梁的惯性矩, 默认为 118.6e-6 m^4.')

parser.add_argument('--area',
                    default=10.3, type=float,
                    help='梁的截面积, 默认为 10.3 m^2.')

parser.add_argument('--load',
                    default=-25000, type=float,
                    help='分布载荷, 默认为 -25000 N/m.')

parser.add_argument('--length',
                    default=10, type=float,
                    help='梁的长度, 默认为 10 m.')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='是否显示进度条日志.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                        help='日志级别, 默认为 INFO.')

parser.add_argument('--beam_type',
                    default='euler_bernoulli_2d', type=str,
                    help='梁的类型, 可选值为 "euler_bernoulli_2d", "normal_2d", "euler_bernoulli_3d".')

parser.add_argument('--pde',
                    default='beam2d', type=str,
                    help='PDE 示例名称, 默认为 "beam2d".')

# 解析参数
options = vars(parser.parse_args())

bm.set_backend('numpy')

beammodel = BeamFEMModel(options)
uh = beammodel.run()
# 打印结果
print("位移解向量 (uh):")
print(uh)
