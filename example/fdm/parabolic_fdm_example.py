import argparse
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.solver import spsolve
from fealpy.fdm import ParabolicFDMModel
from fealpy.utils import timer

tmr = timer()
next(tmr)


## 参数解析
parser = argparse.ArgumentParser(description=
        """ 在均匀网格上使用有限差分方法求解 parabolic 方程 """)

parser.add_argument('--example',
        default='sinsinexp', type=str,
        help="求解的 parabolic方程的算例, 默认是'sinsinexp',其他可选算例还有: 'sinexp', " \
        "还可以用下面的代码可查看可用的算例： \
        from fealpy.model import PDEDataManager \
        PDEDataManager('parabolic').show_examples().")

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--ns',
        default=10, type=int,
        help='初始网格在每个方向剖分段数, 默认 10 段.')

parser.add_argument('--solver',
        default=spsolve,
        help='求解器, 默认是 spsolve')

parser.add_argument('--nt',
        default=400,
        help='时间步的分隔, 默认是 400')

parser.add_argument('--scheme',
        default='backward',
        help="差分格式,默认是 'backward'")

args = parser.parse_args()
example = args.example
ns = args.ns
maxit = args.maxit
solver = args.solver
nt = args.nt
scheme = args.scheme

model = ParabolicFDMModel(example=example, maxit=maxit, ns=ns, solver=solver, 
                           nt=nt, scheme=scheme)
model.run()   # 加密求解过程
tmr.send('Total time')
next(tmr)
model.show_error()  # 误差、误差比的图示
model.show_solution()  # 数值解的图像

