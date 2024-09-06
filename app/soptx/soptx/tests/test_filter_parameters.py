import pytest
import numpy as np
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.sparse import COOTensor
from ..utilfunc.filter_parameters import compute_filter
from .filter_parameters_data import filter_data

from math import ceil, sqrt

class TestFilter:
    @pytest.mark.parametrize("filterdata", filter_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_computer_filter(self, filterdata, backend):
        bm.set_backend(backend)
        nx = filterdata["nx"]
        ny = filterdata["ny"]
        rmin = filterdata["rmin"]
        nfilter = int(nx * ny * ((2 * (ceil(rmin) - 1) + 1) ** 2))
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.zeros(nfilter, dtype=bm.float64)
        cc = 0

        for i in range(nx):
            for j in range(ny):
                row = i * ny + j
                kk1 = int(max(i - (ceil(rmin) - 1), 0))
                kk2 = int(min(i + ceil(rmin), nx))
                ll1 = int(max(j - (ceil(rmin) - 1), 0))
                ll2 = int(min(j + ceil(rmin), ny))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * ny + l
                        fac = rmin - sqrt((i - k) ** 2 + (j - l) ** 2)
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = max(0.0, fac)
                        cc += 1
        indices = bm.astype(bm.stack((iH, jH), axis=0), bm.int32)
        H_coo = COOTensor(indices=bm.astype(bm.stack((iH, jH), axis=0), bm.int32), 
                    values=sH, 
                    spshape=(nx * ny, nx * ny))
        H_csr = COOTensor(indices=bm.astype(bm.stack((iH, jH), axis=0), bm.int32), 
                    values=sH, 
                    spshape=(nx * ny, nx * ny)).tocsr()
        H_coo_dense = H_coo.to_dense()
        H_csr_dense = H_csr.to_dense()

        H, Hs = compute_filter(nx=nelx, ny=nely, rmin=rmin)

        Hs_88_m = filterdata["Hs"]

        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_88_p, decimal=4)
        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_88_m, decimal=4)

