import argparse
from fealpy.backend import backend_manager as bm
from fealpy.fvm import StokesFVMRCModel


def main():
    parser = argparse.ArgumentParser(description="FVM Stokes solver on staggered mesh")

    parser.add_argument('--pde', default=1, type=int,
                        help='PDE example ID from Stokes PDE manager.')
    
    parser.add_argument('--nx', default=40, type=int,
                        help='Number of cells in x-direction.')
    
    parser.add_argument('--ny', default=40, type=int,
                        help='Number of cells in y-direction.')
    
    parser.add_argument('--backend', default='numpy', type=str,
                        help="Backend: numpy, torch, tensorflow, or jax.")
    
    parser.add_argument('--plot', action='store_true',
                        help="Plot numerical solutions.")
    
    parser.add_argument('--log_level', default="INFO", type=str)

    options = vars(parser.parse_args())
    bm.set_backend(options["backend"])

    model = StokesFVMRCModel(options)
    print(model)

    model.solve_rhie_chow()
    uerr, verr, perr = model.compute_error()
    print(f"L2 error (u) = {uerr:.4e}")
    print(f"L2 error (v) = {verr:.4e}")
    print(f"L2 error (p) = {perr:.4e}")
    if options["plot"]:
        model.plot()


if __name__ == "__main__":
    main()
