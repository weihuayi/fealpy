import numpy as np
import pytest

from fealpy.mesh.vtk_reader import read_mesh_from_vtu


class _FakeVtkIdList:
    def __init__(self, ids):
        self._ids = list(ids)

    def GetNumberOfIds(self):
        return len(self._ids)

    def GetId(self, index):
        return self._ids[index]


class _FakeCell:
    def __init__(self, ids):
        self._ids = _FakeVtkIdList(ids)

    def GetPointIds(self):
        return self._ids


class _FakeVtkArray:
    def __init__(self, data, name=None):
        self.data = np.asarray(data)
        self._name = name

    def GetName(self):
        return self._name


class _FakeDataSetAttributes:
    def __init__(self, arrays):
        self._arrays = list(arrays)

    def GetNumberOfArrays(self):
        return len(self._arrays)

    def GetArray(self, index):
        return self._arrays[index]


class _FakePoints:
    def __init__(self, data):
        self._data = _FakeVtkArray(data)

    def GetData(self):
        return self._data


class _FakeCellArray:
    def __init__(self, connectivity, offsets):
        self._connectivity = _FakeVtkArray(connectivity)
        self._offsets = _FakeVtkArray(offsets)

    def GetConnectivityArray(self):
        return self._connectivity

    def GetOffsetsArray(self):
        return self._offsets


class _FakeGrid:
    def __init__(self):
        self.points = _FakePoints(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            )
        )
        self.cells = [
            (5, [0, 1, 2]),
            (5, [1, 3, 2]),
            (3, [0, 1]),
        ]
        self.cell_types = _FakeVtkArray(np.array([cell_type for cell_type, _ in self.cells], dtype=np.uint8))
        connectivity = np.array([point for _, points in self.cells for point in points], dtype=np.int64)
        offsets = np.array([0, 3, 6, 8], dtype=np.int64)
        self.cell_array = _FakeCellArray(connectivity, offsets)
        self.cell_data = _FakeDataSetAttributes([
            _FakeVtkArray(np.array([2, 4, 0]), "marker"),
            _FakeVtkArray(np.array([11, 13, 0]), "RegionId"),
            _FakeVtkArray(np.array([0, 0, 9]), "boundary"),
        ])
        self.point_data = _FakeDataSetAttributes([
            _FakeVtkArray(np.array([10.0, 20.0, 30.0, 40.0]), "temperature"),
        ])

    def GetPoints(self):
        return self.points

    def GetNumberOfCells(self):
        return len(self.cells)

    def GetCellType(self, index):
        return self.cells[index][0]

    def GetCell(self, index):
        raise AssertionError("fast reader should not call GetCell")

    def GetCellTypes(self):
        return self.cell_types

    def GetCellTypesArray(self):
        return self.cell_types

    def GetCells(self):
        return self.cell_array

    def GetCellData(self):
        return self.cell_data

    def GetPointData(self):
        return self.point_data


class _FakeReader:
    def __init__(self):
        self.filename = None
        self.updated = False

    def SetFileName(self, filename):
        self.filename = filename

    def Update(self):
        self.updated = True

    def GetOutput(self):
        return _FakeGrid()


@pytest.fixture
def fake_vtk_modules(monkeypatch):
    from fealpy.mesh import vtk_reader as vr

    fake_vtk = type("FakeVtkModule", (), {
        "VTK_VERTEX": 1,
        "VTK_LINE": 3,
        "VTK_TRIANGLE": 5,
        "VTK_QUAD": 9,
        "VTK_TETRA": 10,
        "VTK_WEDGE": 13,
        "VTK_PYRAMID": 14,
        "VTK_HEXAHEDRON": 12,
        "vtkXMLUnstructuredGridReader": _FakeReader,
    })
    fake_vnp = type("FakeNumpySupport", (), {
        "vtk_to_numpy": staticmethod(lambda arr: np.asarray(arr.data)),
    })

    monkeypatch.setattr(vr, "_load_vtk", lambda: (fake_vtk, fake_vnp))


def test_read_mesh_from_vtu_rebuilds_grouped_entities_and_attributes(fake_vtk_modules):
    mesh = read_mesh_from_vtu("dummy.vtu")

    assert mesh.geo_dimension() == 2
    assert list(mesh.block.root_entity_names) == ["tri"]
    np.testing.assert_allclose(
        mesh.block.positions,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )
    np.testing.assert_array_equal(mesh.block.get_sector("tri").indices, np.array([[0, 1, 2], [1, 3, 2]]))
    np.testing.assert_array_equal(mesh.block.get_sector("segment").indices, np.array([[0, 1]]))
    np.testing.assert_array_equal(mesh.block.get_sector("tri").attributes["marker"], np.array([2, 4]))
    np.testing.assert_array_equal(mesh.block.get_sector("tri").attributes["RegionId"], np.array([11, 13]))
    np.testing.assert_array_equal(mesh.block.get_sector("segment").attributes["boundary"], np.array([9]))
    np.testing.assert_allclose(
        mesh.block.get_sector("point").attributes["temperature"],
        np.array([10.0, 20.0, 30.0, 40.0]),
    )


def test_read_mesh_from_vtu_can_keep_explicit_geometric_dimension(fake_vtk_modules):
    mesh = read_mesh_from_vtu("dummy.vtu", geometric_dimension=3)

    assert mesh.geo_dimension() == 3
    np.testing.assert_allclose(mesh.block.positions[:, 2], np.zeros(4))
