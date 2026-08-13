import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from fealpy.interop.abaqus import read_inp


def test_read_inp_mesh_can_build_surface_and_plot(tmp_path):
    source = tmp_path / "one_tet.inp"
    source.write_text(
        """*Part, name=PART-1
*Node
10, 0.0, 0.0, 0.0
30, 1.0, 0.0, 0.0
40, 0.0, 1.0, 0.0
50, 0.0, 0.0, 1.0
*Element, type=C3D4
100, 10, 30, 40, 50
*End Part
""",
        encoding="utf-8",
    )

    mesh = read_inp(source)

    assert set(mesh.block.sectors) == {"node", "tet"}
    assert mesh.block.relations == {}

    mesh.construct(exclude=["segment", "node"])

    assert mesh.block.get_sector("tri").indices.shape == (4, 3)
    assert mesh.block.relations[("tet", "tri")].tgt_indices.shape == (1, 4)
    boundary_triangles = mesh.Entity("tri").boundary().index
    assert boundary_triangles.shape == (4,)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    artists = mesh.add_plot(
        axes,
        entity="tri",
        index=boundary_triangles,
        alpha=0.5,
    )
    output = tmp_path / "one_tet.png"
    figure.savefig(output)
    plt.close(figure)

    assert len(artists) == 1
    assert output.stat().st_size > 0
