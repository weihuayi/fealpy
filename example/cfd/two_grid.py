from fealpy.backend import backend_manager as bm
from fealpy.cfd import StationaryIncompressibleNSLFEMModel
from fealpy.cfd import TwoGridModel
from fealpy.cfd.equation import 
from fealpy.cfd.model.stationary_incompressible_navier_stokes_2d import FromSympy

fine_model = StationaryIncompressibleNSLFEMModel()
coarse_model = StationaryIncompressibleNSLFEMModel()

model = TwoGridModel(fine_model, coarse_model)
