#!/usr/bin/python3
import argparse

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次拉格朗日有限元方法求解possion方程
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")


args = parser.parse_args()
bm.set_backend(args.backend)

from fealpy.fem import PoissonLFEMModel

model = PoissonLFEMModel()
model.set_pde()
model.set_init_mesh(nx=20, ny=20)
model.set_order()
model.solve.set('cg')
model.run['uniform_refine']()


