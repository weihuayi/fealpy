import argparse
from fealpy.backend import backend_manager as bm

parser = argparse.ArgumentParser(description="FVM Poisson solver with cross-diffusion")

parser.add_argument('--pde', default=3, type=int,
                    help='PDE example ID from Poisson PDE manager')

parser.add_argument('--nx', default=20, type=int, help='Number of cells in x')
parser.add_argument('--ny', default=20, type=int, help='Number of cells in y')

parser.add_argument('--space_degree', default=0, type=int,
                    help='Polynomial degree of ScaledMonomialSpace')

parser.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'],
                    help='Backend type')

parser.add_argument('--pbar_log', default=False, type=bool,
                    help='Show progress bar during execution')

parser.add_argument('--log_level', default='INFO', type=str,
                    help='Logging level: DEBUG, INFO, WARNING, ERROR')

parser.add_argument('--max_iter', default=6, type=int,
                    help='Maximum number of nonlinear iterations')

parser.add_argument('--tol', default=1e-6, type=float,
                    help='Convergence tolerance')

parser.add_argument('--plot', action='store_true',
                    help='Show solution plot after solving')

options = vars(parser.parse_args())
bm.set_backend(options["backend"])
from fealpy.fvm import PoissonFVMModel

model = PoissonFVMModel(options)
model.solve(max_iter=options["max_iter"], tol=options["tol"])
l2_error = model.compute_error()
print(f"L2 error = {l2_error}")
model.plot()


