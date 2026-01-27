import argparse

# 参数解析
parser = argparse.ArgumentParser(description="""
        Solve elastoplasticity problems using the finite element method.
        """)

parser.add_argument('--pde',
                    default=4, type=int,
                    help='Index of the elastoplasticity model, default is 4.')

parser.add_argument('--space_degree',
                    default=2, type=int,
                    help='Polynomial degree for the finite element space.')

parser.add_argument('--device',
                    default='cuda', type=str,
                    help='Computation device, "cpu" or "cuda".')

parser.add_argument('--linear_system',
                    default='matrix_free', type=str,
                    help='Linear system assembly method.')

parser.add_argument('--solver',
                    default='cg', type=str,
                    help='Linear solver to use.')

parser.add_argument('--pbar_log',
                    default=False, action='store_true',
                    help='Show progress bar log.')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Logging level. Default is INFO.')
# 解析参数                                                                                                                                                                                                                  
options = vars(parser.parse_args())
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

from fealpy.csm.fem import PerforatedSquarePlateFEMModel
model = PerforatedSquarePlateFEMModel(options)
pde= model.pde
print()


model.solve()


