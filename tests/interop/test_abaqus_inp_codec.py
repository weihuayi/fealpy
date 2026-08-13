from textwrap import dedent

import pytest

from fealpy.interop.abaqus._codec import (
    AbaqusInpError,
    AbaqusInpSyntaxError,
    UnsupportedAbaqusElementError,
    read_inp_model,
)


def write_inp(tmp_path, content: str):
    path = tmp_path / "model.inp"
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    return path


def test_read_inp_model_parses_noncontiguous_c3d4_part(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Heading
        Small tetrahedron
        ** comments do not start a new section
        *Part, Name=PART-1
        *Node, nset=ALL-NODES
        10, 0.0, 0.0, 0.0
        30, 1.0, 0.0, 0.0
        40, 0.0, 1.0, 0.0
        50, 0.0, 0.0, 1.0
        *Element, type=c3d4, elset=ALL-ELEMENTS, internal
        100, 10, 30, 40, 50
        *Material, name=STEEL
        *Elastic
        210000.0, 0.3
        *End Part
        """,
    )

    model = read_inp_model(path)

    assert len(model.node_blocks) == 1
    assert model.node_blocks[0].part_name == "PART-1"
    assert model.node_blocks[0].options == {"NSET": "ALL-NODES"}
    assert model.node_blocks[0].labels == [10, 30, 40, 50]
    assert model.node_blocks[0].coordinates[-1] == (0.0, 0.0, 1.0)

    assert len(model.element_blocks) == 1
    assert model.element_blocks[0].element_type == "C3D4"
    assert model.element_blocks[0].options == {
        "TYPE": "c3d4",
        "ELSET": "ALL-ELEMENTS",
        "INTERNAL": None,
    }
    assert model.element_blocks[0].labels == [100]
    assert model.element_blocks[0].connectivity == [(10, 30, 40, 50)]

    assert {
        "HEADING",
        "PART",
        "NODE",
        "ELEMENT",
        "MATERIAL",
        "ELASTIC",
        "END PART",
    } <= model.encountered_keywords
    assert model.unmapped_keywords == {"HEADING", "MATERIAL", "ELASTIC"}


def test_multiple_node_and_element_blocks_share_one_part_scope(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Part, name=PART-1
        *Node
        1, 0.0, 0.0, 0.0
        2, 1.0, 0.0, 0.0
        *Node
        3, 0.0, 1.0, 0.0
        4, 0.0, 0.0, 1.0
        5, 1.0, 1.0, 1.0
        *Element, type=C3D4
        10, 1, 2, 3, 4
        *Element, type=C3D4
        20, 2, 3, 4, 5
        *End Part
        """,
    )

    model = read_inp_model(path)

    assert [block.labels for block in model.node_blocks] == [[1, 2], [3, 4, 5]]
    assert [block.labels for block in model.element_blocks] == [[10], [20]]


def test_labels_may_repeat_in_different_parts(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Part, name=LEFT
        *Node
        1, 0.0, 0.0, 0.0
        2, 1.0, 0.0, 0.0
        3, 0.0, 1.0, 0.0
        4, 0.0, 0.0, 1.0
        *Element, type=C3D4
        1, 1, 2, 3, 4
        *End Part
        *Part, name=RIGHT
        *Node
        1, 2.0, 0.0, 0.0
        2, 3.0, 0.0, 0.0
        3, 2.0, 1.0, 0.0
        4, 2.0, 0.0, 1.0
        *Element, type=C3D4
        1, 1, 2, 3, 4
        *End Part
        """,
    )

    model = read_inp_model(path)

    assert [block.part_name for block in model.node_blocks] == ["LEFT", "RIGHT"]
    assert [block.labels for block in model.element_blocks] == [[1], [1]]


@pytest.mark.parametrize(
    ("duplicate_record", "message"),
    [
        ("1, 1.0, 0.0, 0.0", "duplicate node label 1"),
        ("10, 1, 2, 3, 4", "duplicate element label 10"),
    ],
)
def test_duplicate_labels_in_one_part_are_rejected(
    tmp_path,
    duplicate_record,
    message,
):
    node_duplicate = duplicate_record.startswith("1,")
    duplicate_node_line = duplicate_record if node_duplicate else ""
    duplicate_element_line = "" if node_duplicate else duplicate_record
    path = write_inp(
        tmp_path,
        f"""
        *Part, name=PART-1
        *Node
        1, 0.0, 0.0, 0.0
        {duplicate_node_line}
        2, 1.0, 0.0, 0.0
        3, 0.0, 1.0, 0.0
        4, 0.0, 0.0, 1.0
        *Element, type=C3D4
        10, 1, 2, 3, 4
        {duplicate_element_line}
        *End Part
        """,
    )

    with pytest.raises(AbaqusInpSyntaxError, match=message):
        read_inp_model(path)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("1, invalid, 0.0, 0.0", "invalid node coordinates"),
        ("1, 0.0, nan, 0.0", "node coordinates must be finite"),
    ],
)
def test_invalid_node_coordinates_are_rejected(tmp_path, record, message):
    path = write_inp(
        tmp_path,
        f"""
        *Node
        {record}
        """,
    )

    with pytest.raises(AbaqusInpSyntaxError, match=message):
        read_inp_model(path)


def test_malformed_c3d4_record_is_rejected(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Element, type=C3D4
        10, 1, 2, 3
        """,
    )

    with pytest.raises(AbaqusInpSyntaxError, match="four node labels"):
        read_inp_model(path)


def test_unsupported_element_type_is_rejected(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Element, type=C3D8
        10, 1, 2, 3, 4, 5, 6, 7, 8
        """,
    )

    with pytest.raises(UnsupportedAbaqusElementError, match="limited to C3D4"):
        read_inp_model(path)


def test_undefined_node_reference_is_rejected(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Node
        1, 0.0, 0.0, 0.0
        2, 1.0, 0.0, 0.0
        3, 0.0, 1.0, 0.0
        *Element, type=C3D4
        10, 1, 2, 3, 99
        """,
    )

    with pytest.raises(AbaqusInpError, match=r"undefined node labels \[99\]"):
        read_inp_model(path)


def test_unterminated_part_is_rejected(tmp_path):
    path = write_inp(
        tmp_path,
        """
        *Part, name=PART-1
        *Node
        1, 0.0, 0.0, 0.0
        """,
    )

    with pytest.raises(AbaqusInpSyntaxError, match="unterminated Abaqus part"):
        read_inp_model(path)
