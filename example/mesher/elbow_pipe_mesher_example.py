import argparse
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesher import ElbowPipeMesher


def main():
    parser = argparse.ArgumentParser(
        description="Generate an FSI elbow pipe mesh and export VTU files."
    )
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--output_dir", default=".", type=str)

    parser.add_argument("--D", default=25.0, type=float)
    parser.add_argument("--bend_angle", default=90.0, type=float)
    parser.add_argument(
        "--R_bend_inner",
        default=1.5,
        type=float,
        help="Inner bend radius ratio to D.",
    )
    parser.add_argument("--L_in_ratio", default=5.0, type=float)
    parser.add_argument("--L_out_ratio", default=10.0, type=float)
    parser.add_argument("--wall_thickness", default=5.0, type=float)

    parser.add_argument("--mesh_size_global", default=None, type=float)
    parser.add_argument("--mesh_size_bend", default=None, type=float)
    parser.add_argument("--mesh_size_interface", default=None, type=float)

    args = parser.parse_args()
    bm.set_backend(args.backend)

    params = {
        "D": args.D,
        "bend_angle": args.bend_angle,
        "R_bend_inner": args.R_bend_inner,
        "L_in_ratio": args.L_in_ratio,
        "L_out_ratio": args.L_out_ratio,
        "wall_thickness": args.wall_thickness,
        "mesh_size_global": args.mesh_size_global,
        "mesh_size_bend": args.mesh_size_bend,
        "mesh_size_interface": args.mesh_size_interface,
    }

    elbow_mesher = ElbowPipeMesher(params)
    tet_mesh = elbow_mesher.init_mesh()
    mesh_dict = elbow_mesher.mesh_data()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tet_path = out_dir / "elbow_pipe_tetra.vtu"
    tet_mesh.to_vtk(fname=str(tet_path))

    tri_boundary = TriangleMesh(mesh_dict["node"], mesh_dict["boundary_tri"])
    tri_boundary_path = out_dir / "elbow_pipe_boundary_tri.vtu"
    tri_boundary.to_vtk(fname= str(tri_boundary_path))

    tri_interface = TriangleMesh(mesh_dict["node"], mesh_dict["interface_tri"])
    tri_interface_path = out_dir / "elbow_pipe_interface_tri.vtu"
    tri_interface.to_vtk(fname=str(tri_interface_path))

    print(f"tetra vtu: {tet_path}")
    print(f"boundary vtu: {tri_boundary_path}")
    print(f"interface vtu: {tri_interface_path}")


if __name__ == "__main__":
    main()
