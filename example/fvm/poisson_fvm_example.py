import argparse
from fealpy.backend import backend_manager as bm
from fealpy.fvm import PoissonFvmModel

parser = argparse.ArgumentParser(description="FVM Poisson solver with cross-diffusion")

parser.add_argument('--pde', default='SinSin_Sin_Dir_2D', type=str,
                    help='PDE example name from Poisson PDE manager')

parser.add_argument('--nx', default=10, type=int, help='Number of cells in x')
parser.add_argument('--ny', default=10, type=int, help='Number of cells in y')

parser.add_argument('--space_degree', default=0, type=int,
                    help='Polynomial degree of ScaledMonomialSpace')

parser.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'], help='Backend type')
parser.add_argument('--pbar_log', default=False, type=bool, help='Show progress bar')
parser.add_argument('--log_level', default='INFO', type=str, help='Log level')

args = parser.parse_args()
options = vars(args)

bm.set_backend(options["backend"])

model = PoissonFvmModel(options)
uh = model.solution()
l2_error = model.compute_error()

print(f"L2 error = {l2_error}")