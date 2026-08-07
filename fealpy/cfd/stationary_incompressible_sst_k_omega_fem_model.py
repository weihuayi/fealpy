from typing import Union
from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
from fealpy.mesh import Mesh
from fealpy.cfd.equation import (StationaryIncompressibleRANS, 
                            StationaryTurbulentKineticEnergy, 
                            StationarySpecificDissipationRate)

class StationaryIncompressibleSSTKOmegaFEMModel(ComputationalModel):

    def __init__(self, pde, mesh=None, options=None):
        super().__init__(pbar_log = True, log_level="INFO")
        self.options = options
        self.pde = pde
        self.equation_rans = StationaryIncompressibleRANS(pde)
        # self.equation_k = StationaryTurbulentKineticEnergy(pde)
        # self.equation_omega = StationarySpecificDissipationRate(pde)

        self.mesh = mesh

        if options is not None:
            self.solve = options['solve']
            self.run = options['run']
            self.maxstep = options['maxstep']
            self.maxit = options['maxit']
            self.tol = options['tol']

    def method(self):
        from .simulation.fem.stationary_sst_k_omega import (Ossen, 
                                                    StationarySpecificDissipationRatePicard,
                                                    StationaryTurbulentKineticEnergyPicard)
        self.fem_rans = Ossen(self.equation_rans, self.mesh)
        # self.fem_k = StationaryTurbulentKineticEnergyPicard(self.equation_k, self.mesh)
        # self.fem_omega = StationarySpecificDissipationRatePicard(self.equation_omega, self.mesh)

        return self.fem_rans
    
    def linear_system(self):
        BForm = self.fem_rans.BForm()
        LForm = self.fem_rans.LForm()
        return BForm, LForm
    
    def run(self):
        BForm, LForm = self.linear_system()
        # A = BForm.assembly()
        b = LForm.assembly()

    
    



