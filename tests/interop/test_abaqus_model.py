from fealpy.interop.abaqus._model import (
    AbaqusElementBlock,
    AbaqusInpModel,
    AbaqusNodeBlock,
)


def test_model_defaults_are_empty_and_independent():
    first = AbaqusInpModel()
    second = AbaqusInpModel()

    first.encountered_keywords.add("NODE")
    first.node_blocks.append(AbaqusNodeBlock())

    assert second.encountered_keywords == set()
    assert second.node_blocks == []


def test_blocks_preserve_abaqus_labels_and_part_scope():
    nodes = AbaqusNodeBlock(
        part_name="PART-1",
        options={"NSET": "ALL-NODES"},
        labels=[10, 30],
        coordinates=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    )
    elements = AbaqusElementBlock(
        element_type="C3D4",
        part_name="PART-1",
        options={"ELSET": "ALL-ELEMENTS"},
        labels=[100],
        connectivity=[(10, 30, 40, 50)],
    )

    assert nodes.part_name == elements.part_name == "PART-1"
    assert nodes.labels == [10, 30]
    assert elements.element_type == "C3D4"
    assert elements.connectivity == [(10, 30, 40, 50)]


def test_model_distinguishes_encountered_and_unmapped_keywords():
    model = AbaqusInpModel(
        encountered_keywords={"PART", "NODE", "ELEMENT", "MATERIAL"},
        unmapped_keywords={"MATERIAL"},
    )

    assert model.unmapped_keywords <= model.encountered_keywords
    assert "NODE" not in model.unmapped_keywords
