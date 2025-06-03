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

args = parser.parse_args()

E = args.modulus
I = args.inertia
A = args.area
f = args.load
L = args.length

beammodel = BeamFEMModel(example='beam2d')
uh = beammodel.run()
print(uh)
