from fealpy.cgraph import create
from fealpy.cgraph.mesh.hydraulic_pipe import TeePipeMesh
import fealpy.mesher as mesher_module


class DummyTeePipeMesher:
    def __init__(self, params):
        self.params = params

    def init_mesh(self):
        return {"kind": "dummy_tee_mesh", "params": self.params}


def test_tee_pipe_mesh_run(monkeypatch):
    monkeypatch.setattr(mesher_module, "TeePipeMesher", DummyTeePipeMesher)

    params = {
        "D_main": 32.0,
        "D_branch": 32.0,
        "intersect_angle": 90.0,
        "R_fillet": 12.25,
        "L_in_ratio": 5.0,
        "L_out_ratio": 10.0,
        "wall_thickness": 5.0,
        "mesh_size_global": 6.0,
        "mesh_size_junction": 4.0,
        "mesh_size_interface": 3.0,
    }

    node = create("TeePipeMesh")
    result = TeePipeMesh.run(**params)

    assert node is not None
    assert result["kind"] == "dummy_tee_mesh"
    assert result["params"] == params
