import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import StokesFVMStaggeredModel  


def main():
    parser = argparse.ArgumentParser(description="Staggered FVM Stokes Solver")

    parser.add_argument('--pde', default=1, type=int,
                        help='Stokes PDE example ID')
    parser.add_argument('--nx', default=20, type=int,
                        help='Grid divisions in x-direction')
    parser.add_argument('--ny', default=20, type=int,
                        help='Grid divisions in y-direction')
    parser.add_argument('--backend', default='numpy', choices=['numpy', 'cupy'],
                        help='Backend engine')
    parser.add_argument('--pbar_log', action='store_true')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--max_iter', default=100, type=int)
    parser.add_argument('--tol', default=1e-6, type=float)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    options = vars(args)
    bm.set_backend(options["backend"])

    model = StokesFVMStaggeredModel(options)
    print(model)

    model.solve(max_iter=options["max_iter"], tol=options["tol"])
    ue, ve, pe = model.compute_error()
    print(f"Velocity L2 error (u): {ue:.4e}, (v): {ve:.4e}, Pressure L2 error: {pe:.4e}")

    if options["plot"]:
        model.plot()
        model.plot_residual()


if __name__ == "__main__":
    main()
