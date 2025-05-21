import argparse
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.solver import spsolve
from fealpy.fdm import WaveFDMModel
from fealpy.utils import timer

tmr = timer()
next(tmr)

## 参数解析
parser = argparse.ArgumentParser(description=
        """ 在均匀网格上使用有限差分方法求解 wave 方程 """)

parser.add_argument('--example',
        default='sinmix', type=str,
        help="求解的 wave 方程的算例, 默认是'sinmix',其他可选算例还有: 'sincos', " \
        "还可以用下面的代码可查看可用的算例： \
        from fealpy.model import PDEDataManager \
        PDEDataManager('wave').show_examples().")

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--ns',
        default=10, type=int,
        help='初始网格在每个方向剖分段数, 默认 20 段.')

parser.add_argument('--solver',
        default=spsolve,
        help='求解器, 默认是 spsolve')

parser.add_argument('--nt',
        default=400,
        help='时间步的分隔, 默认是 400')

parser.add_argument('--theta',
        default=0.5,
        help=" theta 差分格式,默认是 0.5, 范围是: [0, 1].")

args = parser.parse_args()
example = args.example
ns = args.ns
maxit = args.maxit
solver = args.solver
nt = args.nt
theta = args.theta

model = WaveFDMModel(example=example, maxit=maxit, ns=ns, solver=solver, theta=theta, nt=nt)
model.run()   # 加密求解过程
tmr.send('Total time')
next(tmr)
model.show_error()  # 误差、误差比的图示
model.show_solution()  # 数值解的图像

