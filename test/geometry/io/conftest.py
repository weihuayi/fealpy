"""pytest fixtures for geometry/io tests — generate sample DXF/STEP files."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sample_dxf_path(tmp_path_factory):
    """Generate a minimal DXF file with diverse entity types for testing."""
    import ezdxf

    tmp = tmp_path_factory.mktemp("test_data")
    path = tmp / "sample.dxf"

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Line
    msp.add_line((0, 0), (10, 0))

    # Circle
    msp.add_circle((5, 5), radius=3)

    # Arc
    msp.add_arc((2, 2), radius=4, start_angle=0, end_angle=90)

    # LWPOLYLINE
    msp.add_lwpolyline([(0, 0), (5, 5), (10, 0)], format="xy")

    # Text
    msp.add_text("Hello DXF", dxfattribs={"insert": (1, 1), "height": 0.5})

    # Dimension (linear)
    dimstyle = doc.dimstyles.duplicate_entry("Standard", "TestDim")
    dimstyle.dxf.dimtxt = 0.5
    dimstyle.dxf.dimscale = 1.0
    dimstyle.dxf.dimdec = 2
    msp.add_linear_dim(
        base=(0, -1),
        p1=(0, 0),
        p2=(10, 0),
        location=(5, -2),
        dimstyle="TestDim",
        override={"dimtxsty": "Standard"},
    )

    # Layer colors
    doc.layers.add("DIM", color=3)
    doc.layers.add("CENTER", color=1)
    doc.layers.add("HIDDEN", color=2)
    doc.layers.add("TEXT", color=4)
    doc.layers.add("WALL", color=5)

    # Entities on specific layers
    msp.add_line((0, 20), (10, 20), dxfattribs={"layer": "DIM", "color": 256})
    msp.add_circle((5, 25), radius=2, dxfattribs={"layer": "CENTER", "color": 256})
    msp.add_text("Dim Text", dxfattribs={"layer": "DIM", "insert": (5, 21), "height": 0.4})

    doc.saveas(str(path))
    return str(path)


@pytest.fixture(scope="session")
def sample_step_path(tmp_path_factory):
    """Generate a minimal STEP file (box shape) for testing."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static

    tmp = tmp_path_factory.mktemp("test_data")
    path = tmp / "sample.step"

    box_maker = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0)
    box_maker.Build()
    shape = box_maker.Shape()

    writer = STEPControl_Writer()
    Interface_Static.SetIVal_s("write.step.schema", 3)  # AP214IS
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(path))

    return str(path)
