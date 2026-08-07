import numpy as np

from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view.mesh import Mesh
from fealpy.mesh import vtk_writter as vw


class _FakeVtkArray:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.name = None

    def SetName(self, name: str) -> None:
        self.name = name


class _FakeDataSetAttributes:
    def __init__(self):
        self.arrays = []

    def AddArray(self, arr: _FakeVtkArray) -> None:
        self.arrays.append(arr)


class _FakePoints:
    def __init__(self):
        self.data = None

    def SetData(self, data):
        self.data = data


class _FakeGrid:
    def __init__(self):
        self.points = None
        self.cells = []
        self._cell_data = _FakeDataSetAttributes()
        self._point_data = _FakeDataSetAttributes()

    def SetPoints(self, points):
        self.points = points

    def InsertNextCell(self, cell_type: int, npts: int, node_ids):
        self.cells.append((cell_type, npts, np.asarray(node_ids)))

    def GetCellData(self):
        return self._cell_data

    def GetPointData(self):
        return self._point_data


class _FakeWriter:
    last_instance = None

    def __init__(self):
        _FakeWriter.last_instance = self
        self.filename = None
        self.mode = None
        self.grid = None

    def SetFileName(self, filename: str):
        self.filename = filename

    def SetDataModeToBinary(self):
        self.mode = "binary"

    def SetDataModeToAscii(self):
        self.mode = "ascii"

    def SetInputData(self, grid):
        self.grid = grid

    def Write(self):
        return 1


def _build_mesh() -> Mesh:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    node_block = EntitySector(
        schema_name="point",
        indices=np.array([0, 1, 2], dtype=np.int64),
        attributes={"temperature": np.array([10.0, 20.0, 30.0], dtype=float)},
    )
    tri_block = EntitySector(
        schema_name="tri",
        indices=np.array([[0, 1, 2]], dtype=np.int64),
        attributes={"material": np.array([7], dtype=np.int64)},
    )

    storage = MeshBlock(positions=positions)
    storage.add_sector(node_block)
    storage.add_sector(tri_block)
    return Mesh(storage)


def test_write_mesh_to_vtu_puts_node_attributes_in_point_data(monkeypatch) -> None:
    fake_vtk = type("FakeVtkModule", (), {
        "VTK_VERTEX": 1,
        "VTK_LINE": 3,
        "VTK_TRIANGLE": 5,
        "VTK_QUAD": 9,
        "VTK_TETRA": 10,
        "VTK_WEDGE": 13,
        "VTK_PYRAMID": 14,
        "VTK_HEXAHEDRON": 12,
        "vtkPoints": _FakePoints,
        "vtkUnstructuredGrid": _FakeGrid,
        "vtkXMLUnstructuredGridWriter": _FakeWriter,
    })
    fake_vnp = type("FakeNumpySupport", (), {
        "numpy_to_vtk": staticmethod(lambda arr: _FakeVtkArray(arr)),
    })

    def _fake_import(name: str):
        if name == "vtk":
            return fake_vtk
        if name == "vtk.util.numpy_support":
            return fake_vnp
        raise ImportError(name)

    monkeypatch.setattr(vw.importlib, "import_module", _fake_import)

    mesh = _build_mesh()
    vw.write_mesh_to_vtu("dummy.vtu", mesh, binary=False)

    grid = _FakeWriter.last_instance.grid
    point_data = {arr.name: arr.data for arr in grid.GetPointData().arrays}
    cell_data = {arr.name: arr.data for arr in grid.GetCellData().arrays}

    assert "temperature" in point_data
    np.testing.assert_allclose(point_data["temperature"], np.array([10.0, 20.0, 30.0]))

    assert "material" in cell_data
    np.testing.assert_array_equal(cell_data["material"], np.array([0, 0, 0, 7]))

    assert "temperature" not in cell_data
