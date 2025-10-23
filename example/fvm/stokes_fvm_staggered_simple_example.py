import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import StokesFVMStaggeredSimpleModel  


def main():
    parser = argparse.ArgumentParser(description="Staggered FVM Stokes Solver")

    parser.add_argument('--pde', default=1, type=int,
                        help='Stokes PDE example ID')
    
    parser.add_argument('--nx', default=20, type=int,
                        help='Grid divisions in x-direction')

    parser.add_argument('--ny', default=20, type=int,
                        help='Grid divisions in y-direction')
    
    parser.add_argument('--backend',default='numpy', type=str,
                        help="the backend of fealpy, can be 'numpy', 'torch', 'tensorflow' or 'jax'.")
    
    parser.add_argument('--pbar_log', default=True, type=bool,
                        help='Whether to show progress bar, default is True')
    
    parser.add_argument('--log_level',
                        default='INFO', type=str,
                        help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

    parser.add_argument('--max_iter', default=1000, type=int)

    parser.add_argument('--tol', default=1e-5, type=float)

    parser.add_argument('--plot', action='store_true')

    options = vars(parser.parse_args())
    
    bm.set_backend(options["backend"])

    model = StokesFVMStaggeredSimpleModel(options)
    print(model)

    model.solve(max_iter=options["max_iter"], tol=options["tol"])
    uerror, verror, perror = model.compute_error()
    print(f"L2 error (u) = {uerror}")
    print(f"L2 error (v) = {verror}")
    print(f"L2 error (p) = {perror}")
    model.plot()
    if options["plot"]:
        model.plot()
        model.plot_residual()


if __name__ == "__main__":
    main()
