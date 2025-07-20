#!/usr/bin/python3
import argparse

from fealpy.backend import backend_manager as bm

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次拉格朗日有限元方法求解椭圆界面问题
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")

parser.add_argument('--pde',
                    default='1', type=str,
                    help='Idx of the PDE model, default is 1')

parser.add_argument('--init_mesh',
                    default='uniform_tri', type=str,
                    help='Type of mesh, default is uniform_tri')

parser.add_argument('--space_degree',
        default=1, type=int,
        help='Degree of Lagrange finite element space, default is 1')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem import InterfacePoissonLFEMModel

model = InterfacePoissonLFEMModel(options)
model.solve.set('cg')
model.run['uniform_refine']()