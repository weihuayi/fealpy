"""Import and display the Case3 Abaqus C3D4 mesh.

Run interactively from the repository root:

    python example/interop/abaqus_case3_import.py

Render without opening a GUI:

    python example/interop/abaqus_case3_import.py \
        --no-show --output /tmp/abaqus_case3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "case3" / "box_case3.inp"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the Case3 Abaqus C3D4 mesh and plot its boundary.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Abaqus INP file (default: data/case3/box_case3.inp)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional image output path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open an interactive plotting window",
    )
    args = parser.parse_args()
    if args.no_show and args.output is None:
        parser.error("--no-show requires --output so the rendering is observable")
    return args


def main() -> None:
    args = parse_arguments()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Abaqus INP file not found: {input_path}\n"
            "Provide it with --input if Case3 is stored elsewhere."
        )

    from fealpy.interop.abaqus import read_inp

    started = perf_counter()
    mesh = read_inp(input_path)
    import_seconds = perf_counter() - started

    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    NF = mesh.number_of_faces()

    print(f"Imported {NN} nodes and {NC} C3D4 elements")
    print(f"Import time: {import_seconds:.3f} s")
    print("Support coverage: C3D4 mesh geometry only")

    started = perf_counter()
    mesh.construct(exclude=["segment", "node"])
    face = mesh.entity("face")
    bdindex = mesh.boundary_face_index()
    topology_seconds = perf_counter() - started

    NBF = int(bdindex.shape[0])
    print(
        f"Constructed {NF} unique triangles; "
        f"selected {NBF} boundary triangles"
    )
    print(f"Surface topology time: {topology_seconds:.3f} s")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    started = perf_counter()
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    mesh.add_plot(
        axes,
        entity="face",
        index=bdindex,
        edgecolor="#315B8A",
        cellcolor="#5B8FF9",
        linewidths=0.05,
        alpha=0.9,
        showaxis=True,
    )
    axes.set_title("FEALPy | Abaqus Case3 boundary mesh")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Wrote image: {output_path}")

    render_seconds = perf_counter() - started
    print(f"Render time: {render_seconds:.3f} s")

    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
