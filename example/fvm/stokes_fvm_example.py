import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import StokesFVMModel

def main():
    parser = argparse.ArgumentParser(description="SIMPLE-based FVM Stokes Solver")

    parser.add_argument('--pde', default=1, type=int, help='Stokes PDE example ID')
    parser.add_argument('--nx', default=20, type=int, help='Grid divisions in x')
    parser.add_argument('--ny', default=20, type=int, help='Grid divisions in y')
    parser.add_argument('--space_degree', default=0, type=int, help='Space degree')
    parser.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'], help='Backend engine')
    parser.add_argument('--pbar_log', action='store_true')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--max_iter', default=100, type=int)
    parser.add_argument('--tol', default=1e-5, type=float)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    options = vars(args)

    bm.set_backend(options["backend"])

    model = StokesFVMModel(options)
    print(model)

    model.solve(max_iter=options["max_iter"], tol=options["tol"])
    Verror, Perror = model.compute_error()
    print(f"Velocity L2 error = {Verror:.4e}")
    print(f"Pressure L2 error = {Perror:.4e}")

    if options["plot"]:
        model.plot()
        model.plot_residual()


if __name__ == "__main__":
    main()
