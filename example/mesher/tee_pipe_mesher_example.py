import argparse
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesher import TeePipeMesher


def main():
    parser = argparse.ArgumentParser(
        description="Generate an FSI tee pipe mesh and export VTU files."
    )
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--output_dir", default=".", type=str)

    parser.add_argument("--D_main", default=32.0, type=float)
    parser.add_argument("--D_branch", default=32.0, type=float)
    parser.add_argument("--intersect_angle", default=90.0, type=float)
    parser.add_argument("--R_fillet", default=12.25, type=float)
    parser.add_argument("--L_in_ratio", default=5.0, type=float)
    parser.add_argument("--L_out_ratio", default=10.0, type=float)
    parser.add_argument("--wall_thickness", default=5.0, type=float)

    parser.add_argument("--mesh_size_global", default=6.0, type=float)
    parser.add_argument("--mesh_size_junction", default=4.0, type=float)
    parser.add_argument("--mesh_size_interface", default=3.0, type=float)

    args = parser.parse_args()
    bm.set_backend(args.backend)

    params = {
        "D_main": args.D_main,
        "D_branch": args.D_branch,
        "intersect_angle": args.intersect_angle,
        "R_fillet": args.R_fillet,
        "L_in_ratio": args.L_in_ratio,
        "L_out_ratio": args.L_out_ratio,
        "wall_thickness": args.wall_thickness,
        "mesh_size_global": args.mesh_size_global,
        "mesh_size_junction": args.mesh_size_junction,
        "mesh_size_interface": args.mesh_size_interface,
    }

    tee_mesher = TeePipeMesher(params)
    tet_mesh = tee_mesher.init_mesh()
    mesh_dict = tee_mesher.mesh_data()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tet_path = out_dir / "tee_pipe_tetra.vtu"
    tet_mesh.to_vtk(fname=str(tet_path))

    tri_boundary = TriangleMesh(mesh_dict["node"], mesh_dict["boundary_tri"])
    tri_boundary_path = out_dir / "tee_pipe_boundary_tri.vtu"
    tri_boundary.to_vtk(fname=str(tri_boundary_path))

    tri_interface = TriangleMesh(mesh_dict["node"], mesh_dict["interface_tri"])
    tri_interface_path = out_dir / "tee_pipe_interface_tri.vtu"
    tri_interface.to_vtk(fname=str(tri_interface_path))

    print(f"tetra vtu: {tet_path}")
    print(f"boundary vtu: {tri_boundary_path}")
    print(f"interface vtu: {tri_interface_path}")


if __name__ == "__main__":
    main()
