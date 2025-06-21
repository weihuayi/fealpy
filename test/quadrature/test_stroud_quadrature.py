import pytest

from fealpy.backend import backend_manager as bm
from fealpy.quadrature.stroud_quadrature import StroudQuadrature 

from stroud_quadrature_data import *


class TestStroudQuadrature:
    @pytest.mark.parametrize('backend',['numpy', 'pytorch'])
    @pytest.mark.parametrize('data', data)
    def test_stroud_quadrature(self, backend, data):
        bm.set_backend(backend)
        dim = data['dim']
        p = data['p']
        qf = StroudQuadrature(dim, p)
        bcs, ws = qf.get_quadrature_points_and_weights()
        np.testing.assert_allclose(bcs, data['bcs'])
        np.testing.assert_allclose(ws, data['ws'])

