import numpy as np
import pytest

from fealpy.backend import bm
from fealpy.interop.abaqus._adapter import (
    AbaqusMeshAdapterError,
    abaqus_model_to_mesh,
)
from fealpy.interop.abaqus._codec import read_inp_model
from fealpy.interop.abaqus._model import (
    AbaqusElementBlock,
    AbaqusInpModel,
    AbaqusNodeBlock,
)


def to_numpy(value):
    return bm.to_numpy(value)


def make_part_model(part_name="PART-1"):
    return AbaqusInpModel(
        node_blocks=[
            AbaqusNodeBlock(
                part_name=part_name,
                labels=[30, 10],
                coordinates=[(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            ),
            AbaqusNodeBlock(
                part_name=part_name,
                labels=[50, 40, 70],
                coordinates=[
                    (0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0),
                ],
            ),
        ],
        element_blocks=[
            AbaqusElementBlock(
                element_type="C3D4",
                part_name=part_name,
                labels=[200],
                connectivity=[(10, 30, 40, 50)],
            ),
            AbaqusElementBlock(
                element_type="C3D4",
                part_name=part_name,
                labels=[100],
                connectivity=[(30, 40, 50, 70)],
            ),
        ],
    )


def test_codec_and_adapter_form_a_complete_mesh_conversion_boundary(tmp_path):
    path = tmp_path / "one_tet.inp"
    path.write_text(
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

    mesh = abaqus_model_to_mesh(read_inp_model(path))

    assert mesh.block.positions.shape == (4, 3)
    np.testing.assert_array_equal(
        to_numpy(mesh.block.get_sector("tet").indices),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )


def test_adapter_maps_labels_and_merges_blocks_without_topology():
    mesh = abaqus_model_to_mesh(make_part_model())

    np.testing.assert_allclose(
        to_numpy(mesh.block.positions),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        to_numpy(mesh.block.get_sector("node").indices),
        np.arange(5, dtype=np.int64).reshape(-1, 1),
    )
    np.testing.assert_array_equal(
        to_numpy(mesh.block.get_sector("tet").indices),
        np.array([[1, 0, 3, 2], [0, 3, 2, 4]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        to_numpy(mesh.block.get_sector("node").attributes["abaqus_label"]),
        np.array([30, 10, 50, 40, 70], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        to_numpy(mesh.block.get_sector("tet").attributes["abaqus_label"]),
        np.array([200, 100], dtype=np.int64),
    )
    assert mesh.block.root_entity_names == ["tet"]
    assert mesh.block.relations == {}
    assert set(mesh.block.sectors) == {"node", "tet"}


def test_adapter_selects_an_explicit_part_without_assembling_parts():
    left = make_part_model("LEFT")
    right = make_part_model("RIGHT")
    model = AbaqusInpModel(
        node_blocks=left.node_blocks + right.node_blocks,
        element_blocks=left.element_blocks + right.element_blocks,
    )

    with pytest.raises(AbaqusMeshAdapterError, match="multiple mesh scopes"):
        abaqus_model_to_mesh(model)

    mesh = abaqus_model_to_mesh(model, part_name="RIGHT")

    assert mesh.block.positions.shape == (5, 3)
    assert mesh.block.get_sector("tet").indices.shape == (2, 4)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (AbaqusInpModel(), "contains no mesh blocks"),
        (
            AbaqusInpModel(
                element_blocks=[
                    AbaqusElementBlock(
                        element_type="C3D4",
                        part_name="PART-1",
                        labels=[1],
                        connectivity=[(1, 2, 3, 4)],
                    )
                ]
            ),
            "contains no nodes",
        ),
        (
            AbaqusInpModel(
                node_blocks=[
                    AbaqusNodeBlock(
                        part_name="PART-1",
                        labels=[1],
                        coordinates=[(0.0, 0.0, 0.0)],
                    )
                ]
            ),
            "contains no C3D4 elements",
        ),
    ],
)
def test_adapter_rejects_incomplete_mesh_scopes(model, message):
    with pytest.raises(AbaqusMeshAdapterError, match=message):
        abaqus_model_to_mesh(model)


def test_adapter_rejects_undefined_node_references():
    model = make_part_model()
    model.element_blocks[0].connectivity[0] = (10, 30, 40, 999)

    with pytest.raises(AbaqusMeshAdapterError, match=r"node labels \[999\]"):
        abaqus_model_to_mesh(model)


def test_adapter_rejects_non_three_dimensional_c3d4_nodes():
    model = make_part_model()
    model.node_blocks[0].coordinates[0] = (1.0, 0.0)

    with pytest.raises(AbaqusMeshAdapterError, match="requires three coordinates"):
        abaqus_model_to_mesh(model)


def test_adapter_rejects_duplicate_labels_in_manual_models():
    model = make_part_model()
    model.node_blocks[1].labels[0] = 30

    with pytest.raises(AbaqusMeshAdapterError, match="duplicate node label 30"):
        abaqus_model_to_mesh(model)
