import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm.experimental import NSFVMStaggeredPISOModel


def main():
    parser = argparse.ArgumentParser(
        description="Staggered PISO-based FVM Navier-Stokes solver"
    )

    parser.add_argument("--pde", default=3, type=int,
                        help="Navier-Stokes PDE example ID")

    parser.add_argument("--nx", default=20, type=int,
                        help="Grid divisions in x")

    parser.add_argument("--ny", default=20, type=int,
                        help="Grid divisions in y")

    parser.add_argument("--nt", default=20, type=int,
                        help="Number of time steps")

    parser.add_argument("--duration", nargs=2, default=(0.0, 1.0), type=float,
                        help="Start and end time")

    parser.add_argument("--backend", default="numpy", type=str,
                        help="Backend: numpy, torch, tensorflow, or jax.")

    parser.add_argument("--pbar_log", default=True, type=bool,
                        help="Whether to show progress bar, default is True")

    parser.add_argument("--log_level", default="INFO", type=str,
                        help="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    parser.add_argument("--plot", action="store_true")

    options = vars(parser.parse_args())

    bm.set_backend(options["backend"])

    model = NSFVMStaggeredPISOModel(options)
    print(model)

    model.solve()
    uerror, verror, perror = model.compute_error()
    print(f"L2 error (u) = {uerror}")
    print(f"L2 error (v) = {verror}")
    print(f"L2 error (p) = {perror}")
    # model.plot()
    if options["plot"]:
        model.plot()


if __name__ == "__main__":
    main()
