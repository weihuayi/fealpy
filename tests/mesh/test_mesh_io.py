import meshio
import numpy as np

from fealpy.mesh.mesh_io import read, write


def test_read_write_mixed_cell_sectors(tmp_path):
    source = tmp_path / "mixed.vtu"
    target = tmp_path / "selected.vtu"
    meshio.write(
        source,
        meshio.Mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            cells=[
                ("triangle", np.array([[0, 1, 2]], dtype=np.int32)),
                ("quad", np.array([[0, 1, 2, 3]], dtype=np.int32)),
            ],
        ),
    )

    block = read(source, file_format="vtu")
    assert set(block.root_entity_names) == {"tri", "quad"}

    write(target, block, ["tri", "quad"], file_format="vtu")
    result = meshio.read(target, file_format="vtu")
    assert {cell.type for cell in result.cells} == {"triangle", "quad"}
