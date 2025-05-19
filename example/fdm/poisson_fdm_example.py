import argparse
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.solver import spsolve
from fealpy.fdm.poisson_fdm_model import PoissonFDMModel


## 参数解析
parser = argparse.ArgumentParser(description=
        """ 在均匀网格上使用有限差分方法求解 poisson 方程 """)

parser.add_argument('--example',
        default='sinsin', type=str,
        help="求解的 poisson 方程的算例, 默认是'sinsin',其他可选算例还有: 'sin', 'coscso', " \
        "还可以用下面的代码可查看可用的算例： \
        from fealpy.model import PDEDataManager \
        PDEDataManager('poisson').show_examples().")

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--ns',
        default=20, type=int,
        help='初始网格在每个方向剖分段数, 默认 20 段.')

parser.add_argument('--solver',
        default=spsolve,
        help='求解器, 默认是 spsolve')

args = parser.parse_args()
example = args.example
ns = args.ns
maxit = args.maxit
solver = args.solver

model = PoissonFDMModel(example=example, maxit=maxit, ns=ns, solver=solver)
model.run()   # 加密求解过程
model.show_error()  # 误差、误差比的图示
model.show_solution()  # 数值解的图像
plt.show()