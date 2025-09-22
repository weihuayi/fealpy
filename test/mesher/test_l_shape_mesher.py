import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.mesher import LshapeMesher

from l_shape_mesher_data import l_shape_mesher_data


class TestLshapeMesher:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", l_shape_mesher_data)
    def test_model_init(self, meshdata, backend):
        bm.set_backend(backend)
        big_box = meshdata['big_box']
        small_box = meshdata['small_box']
        nx = meshdata['nx']
        ny = meshdata['ny']
        mesh_type = meshdata['mesh_type']
        l_shape_mesher = LshapeMesher()
        l_shape_mesh = l_shape_mesher.init_mesh[mesh_type](big_box,
                                                           small_box,
                                                           nx=nx, ny=ny)

