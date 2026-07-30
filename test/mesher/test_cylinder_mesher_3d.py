import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesher import CylinderMesher


def test_cylinder_mesher_builds_current_tetrahedron_mesh():
    pytest.importorskip("gmsh")
    bm.set_backend("numpy")
    mesher = CylinderMesher(radius=0.5, height=3.0, lc=0.4)

    mesh = mesher.init_mesh["tet"]()

    assert mesh.geo_dimension() == 3
    assert mesh.number_of_cells() > 0
    points = np.asarray(bm.to_numpy(mesh.entity("node")), dtype=float)
    assert np.max(points[:, 0] ** 2 + points[:, 1] ** 2) <= 0.25 + 1.0e-12
    assert points[:, 2].min() >= -1.0e-12
    assert points[:, 2].max() <= 3.0 + 1.0e-12
    volume = np.asarray(bm.to_numpy(mesh.entity_measure("cell")), dtype=float).sum()
    assert volume == pytest.approx(0.75 * np.pi, rel=0.08)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"radius": 0.0, "height": 3.0, "lc": 0.2}, "radius"),
        ({"radius": 0.5, "height": 0.0, "lc": 0.2}, "height"),
        ({"radius": 0.5, "height": 3.0, "lc": 0.0}, "lc"),
    ],
)
def test_cylinder_mesher_rejects_nonpositive_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CylinderMesher(**kwargs)
