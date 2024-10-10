import pytest
import numpy as np
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.sparse import COOTensor
from ..utilfs.filter_parameters import compute_filter, compute_filter_3d, apply_filter
from .filter_parameters_data import *

from math import ceil, sqrt

class TestFilter:
    @pytest.mark.parametrize("filterdata", filter_data_2d)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_computer_filter(self, filterdata, backend):
        bm.set_backend(backend)
        nx = filterdata["nx"]
        ny = filterdata["ny"]
        rmin = filterdata["rmin"]

        H, Hs = compute_filter(nx=nx, ny=ny, rmin=rmin)

        Hs_true = filterdata["Hs"]

        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_true, decimal=4)

    @pytest.mark.parametrize("filterdata", filter_data_2d)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_apply_filter(self, filterdata, backend):
        bm.set_backend(backend)

        dce = bm.copy(bm.from_numpy(filterdata["dce"]))
        dve = bm.copy(bm.from_numpy(filterdata["dve"]))
        rho = bm.from_numpy(filterdata["rho"])
        nx = filterdata["nx"]
        ny = filterdata["ny"]
        rmin = filterdata["rmin"]

        H, Hs = compute_filter(nx=nx, ny=ny, rmin=rmin)
        dce_sens_updated, _ = apply_filter(
            ft=0, rho=rho,
            dce=dce, dve=dve,
            H=H, Hs=Hs)
        dce_sens_updated_true = filterdata['dce_sens_updated']

        np.testing.assert_almost_equal(bm.to_numpy(dce_sens_updated),
                                       dce_sens_updated_true, decimal=4)

    @pytest.mark.parametrize("filterdata", filter_data_3d)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_computer_filter(self, filterdata, backend):
        bm.set_backend(backend)
        nx = filterdata["nx"]
        ny = filterdata["ny"]
        nz = filterdata["nz"]
        rmin = filterdata["rmin"]

        H, Hs = compute_filter_3d(nx=nx, ny=ny, nz=nz, rmin=rmin)

        Hs_true = filterdata["Hs"]

        np.testing.assert_almost_equal(bm.to_numpy(Hs), Hs_true, decimal=4)
    



