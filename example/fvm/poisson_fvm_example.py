import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import PoissonFVMModel


def main():
    parser = argparse.ArgumentParser(description="FVM Poisson solver with cross-diffusion")

    parser.add_argument('--pde', default=3, type=int,
                        help='PDE example ID from Poisson PDE manager.')

    parser.add_argument('--nx', default=20, type=int,
                        help='Number of cells in x-direction.')
    parser.add_argument('--ny', default=20, type=int,
                        help='Number of cells in y-direction.')

    parser.add_argument('--space_degree', default=0, type=int,
                        help='Polynomial degree of ScaledMonomialSpace.')

    parser.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'],
                        help='Backend type: numpy or cupy.')

    parser.add_argument('--pbar_log', action='store_true',
                        help='Enable progress bar during execution.')
    parser.add_argument('--log_level', default='INFO', type=str,
                        help='Logging level: DEBUG, INFO, WARNING, or ERROR.')

    parser.add_argument('--max_iter', default=6, type=int,
                        help='Maximum number of nonlinear iterations.')
    parser.add_argument('--tol', default=1e-6, type=float,
                        help='Convergence tolerance for fixed-point iterations.')

    parser.add_argument('--plot', action='store_true',
                        help='Display solution plots after solving.')

    options = vars(parser.parse_args())

    bm.set_backend(options["backend"])

    model = PoissonFVMModel(options)
    print(model)
    model.solve(max_iter=options["max_iter"], tol=options["tol"])

    l2_error = model.compute_error()
    print(f"L2 error = {l2_error:.4e}")
    
    if options["plot"]:
        model.plot()


if __name__ == "__main__":
    main()
