from pathlib import Path

import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesher import ElbowPipeMesher, ElbowPipeMesher2D, ElbowPipeMesher3D


def test_elbow_pipe_mesher_defaults_to_3d():
    mesher = ElbowPipeMesher()

    assert isinstance(mesher.impl, ElbowPipeMesher3D)
    assert mesher.geo_dimension() == 3


def test_elbow_pipe_mesher_selects_2d():
    mesher = ElbowPipeMesher(dim=2)

    assert isinstance(mesher.impl, ElbowPipeMesher2D)
    assert mesher.geo_dimension() == 2


def test_elbow_pipe_mesher_from_parameters_keeps_dim_selection():
    mesher = ElbowPipeMesher.from_remesher_parameters(dim=2)

    assert isinstance(mesher.impl, ElbowPipeMesher2D)


def _save_2d_mesh_preview(mesh_data, output_path: Path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.tri import Triangulation

    node = bm.to_numpy(mesh_data["node"])
    triangle = bm.to_numpy(mesh_data["triangle"])
    triangle_region = bm.to_numpy(mesh_data["triangle_region"])
    boundary_edge = bm.to_numpy(mesh_data["boundary_edge"])
    boundary_marker = bm.to_numpy(mesh_data["boundary_edge_marker"])
    boundary_edge_index = bm.to_numpy(mesh_data["boundary_edge_index"])
    interface_edge = bm.to_numpy(mesh_data["interface_edge"])
    interface_edge_index = bm.to_numpy(mesh_data["interface_edge_index"])
    name_to_marker = {
        name: dimtag[1]
        for name, dimtag in mesh_data["physical_name_to_dimtag"].items()
    }
    region_styles = {
        1: {"color": "#9ecae1", "label": "fluid"},
        2: {"color": "#f4c7ab", "label": "solid"},
    }
    boundary_styles = {
        "inlet": {"color": "#1f77b4", "linewidth": 1.8, "label": "inlet"},
        "outlet": {"color": "#ff7f0e", "linewidth": 1.8, "label": "outlet"},
        "fsi_interface": {"color": "#c9184a", "linewidth": 2.2, "label": "fsi_interface"},
        "outer_wall": {"color": "#2ca02c", "linewidth": 1.2, "label": "outer_wall"},
        "solid_inlet_end": {"color": "#9467bd", "linewidth": 1.2, "label": "solid_inlet_end"},
        "solid_outlet_end": {"color": "#8c564b", "linewidth": 1.2, "label": "solid_outlet_end"},
    }

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    triangulation = Triangulation(node[:, 0], node[:, 1], triangle)
    for region_id, style in region_styles.items():
        mask = triangle_region == region_id
        if mask.any():
            ax.tripcolor(
                triangulation,
                facecolors=mask.astype(float),
                cmap=matplotlib.colors.ListedColormap(["none", style["color"]]),
                shading="flat",
                alpha=0.55,
                zorder=0,
            )
    ax.triplot(triangulation, color="#d0d0d0", linewidth=0.35, zorder=1)

    legend_handles = [
        Line2D([0], [0], color="#d0d0d0", linewidth=0.8, label="cells"),
        Line2D([0], [0], color=region_styles[1]["color"], linewidth=6, label=region_styles[1]["label"]),
        Line2D([0], [0], color=region_styles[2]["color"], linewidth=6, label=region_styles[2]["label"]),
    ]

    for name, style in boundary_styles.items():
        marker = name_to_marker.get(name)
        if marker is None:
            continue
        mask = boundary_marker == marker
        if not mask.any():
            continue
        segments = node[boundary_edge[mask]]
        ax.add_collection(
            LineCollection(
                segments,
                colors=style["color"],
                linewidths=style["linewidth"],
                alpha=0.9,
                zorder=2,
            )
        )
        legend_handles.append(
            Line2D([0], [0], color=style["color"], linewidth=style["linewidth"], label=style["label"])
        )

    if len(interface_edge):
        ax.add_collection(LineCollection(node[interface_edge], colors="#c9184a", linewidths=1.8, zorder=3))

    ax.scatter(node[:, 0], node[:, 1], s=0.35, color="#222222", alpha=0.25, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("ElbowPipeMesher 2D current mesh state")
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def test_elbow_pipe_mesher_2d_mesh_data_and_visualization():
    params = {
        "mesh_size_global": 5.0,
        "mesh_size_bend": 3.0,
        "mesh_size_interface": 2.5,
    }
    mesher = ElbowPipeMesher(dim=2, params=params)
    mesh = mesher.init_mesh()
    mesh_data = mesher.mesh_data()

    assert isinstance(mesh, TriangleMesh)
    assert mesh_data["node"].shape[1] == 2
    assert mesh_data["triangle"].shape[1] == 3
    assert mesh_data["triangle_region"].shape == (mesh_data["triangle"].shape[0],)
    assert mesh_data["boundary_edge"].shape[1] == 2
    assert mesh_data["boundary_edge_marker"].shape == (mesh_data["boundary_edge"].shape[0],)
    assert mesh_data["boundary_edge_index"].shape == (mesh_data["boundary_edge"].shape[0],)
    assert mesh_data["interface_edge"].shape[1] == 2
    assert mesh_data["interface_adjacent_triangle"].shape == (mesh_data["interface_edge"].shape[0], 2)
    assert mesh_data["interface_adjacent_region"].shape == (mesh_data["interface_edge"].shape[0], 2)
    assert mesh_data["interface_edge_index"].shape == (mesh_data["interface_edge"].shape[0],)

    assert set(mesh_data["physical_name_to_dimtag"]) == {
        "fluid",
        "solid",
        "inlet",
        "outlet",
        "fsi_interface",
        "outer_wall",
        "solid_inlet_end",
        "solid_outlet_end",
    }
    assert set(map(int, mesh_data["triangle_region"].tolist())) == {1, 2}
    assert set(map(int, mesh_data["boundary_edge_marker"].tolist())) == {3, 4, 5, 6, 7, 8}
    assert {tuple(map(int, pair)) for pair in mesh_data["interface_adjacent_region"].tolist()} == {(1, 2)}

    boundary_edge_index = bm.to_numpy(mesh_data["boundary_edge_index"])
    interface_edge_index = bm.to_numpy(mesh_data["interface_edge_index"])
    assert boundary_edge_index.tolist() == sorted(boundary_edge_index.tolist())
    assert interface_edge_index.tolist() == sorted(interface_edge_index.tolist())
    assert bm.to_numpy(mesh.edge)[boundary_edge_index].tolist() == bm.to_numpy(mesh_data["boundary_edge"]).tolist()
    assert bm.to_numpy(mesh.edge)[interface_edge_index].tolist() == bm.to_numpy(mesh_data["interface_edge"]).tolist()

    name_to_marker = {name: dimtag[1] for name, dimtag in mesh_data["physical_name_to_dimtag"].items()}
    boundary_edge = bm.to_numpy(mesh_data["boundary_edge"])
    boundary_marker = bm.to_numpy(mesh_data["boundary_edge_marker"])
    solid_inlet = bm.to_numpy(mesh_data["node"])[boundary_edge[boundary_marker == name_to_marker["solid_inlet_end"]]]
    solid_outlet = bm.to_numpy(mesh_data["node"])[boundary_edge[boundary_marker == name_to_marker["solid_outlet_end"]]]
    boundary_index = bm.to_numpy(mesh_data["boundary_edge_index"])
    interface_index = bm.to_numpy(mesh_data["interface_edge_index"])
    assert solid_inlet[:, :, 0].max() - solid_inlet[:, :, 0].min() < 1e-12
    assert solid_inlet[:, :, 1].min() < -10.0
    assert solid_inlet[:, :, 1].max() > 10.0
    assert solid_outlet[:, :, 1].max() - solid_outlet[:, :, 1].min() < 1e-12
    assert solid_outlet[:, :, 0].min() < 40.0
    assert solid_outlet[:, :, 0].max() > 60.0
    assert mesh_data["triangle"].shape[0] > 2500
    assert mesh_data["boundary_edge"][bm.to_numpy(mesh_data["boundary_edge_marker"]) == name_to_marker["outer_wall"]].shape[0] > 150
    assert boundary_index.tolist() == sorted(boundary_index.tolist())
    assert interface_index.tolist() == sorted(interface_index.tolist())
    assert bm.to_numpy(mesh.edge)[boundary_index].tolist() == bm.to_numpy(mesh_data["boundary_edge"]).tolist()
    assert bm.to_numpy(mesh.edge)[interface_index].tolist() == bm.to_numpy(mesh_data["interface_edge"]).tolist()

    mesh_region = bm.to_numpy(mesh.celldata["region"])
    triangle_region = bm.to_numpy(mesh_data["triangle_region"])
    assert mesh_region.shape == triangle_region.shape
    assert mesh_region.tolist() == triangle_region.tolist()

    preview_path = Path(__file__).with_name("elbow_pipe_2d_preview.png")
    _save_2d_mesh_preview(mesh_data, preview_path)
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0
