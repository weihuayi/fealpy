import pytest
import numpy as np
from fealpy.experimental.backend import backend_manager as bm

from ..utilfunc.filter_parameters import compute_filter

from .filter_parameters_data import filter_data
from scipy.sparse import coo_matrix

class TestFilter:
    @pytest.mark.parametrize("filterdata", filter_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_coomputer_filter(self, filterdata, backend):
        bm.set_backend(backend)
        nelx = 6
        nely = 3
        rmin = 1.5
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt(((i-k) * (i-k) + (j-l) * (j-l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        H_88_p = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
        Hs_88_p = H_88_p.sum(1).A.flatten()

        H, Hs = compute_filter(nx=nelx, ny=nely, rmin=rmin)

        Hs_88_m = filterdata["Hs"]

        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_88_p, decimal=4)
        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_88_m, decimal=4)
