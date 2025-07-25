from fealpy.backend import backend_manager as bm
from ..model import ComputationalModel

class TwoGridModel(ComputationalModel):
    def __init__(self, coarsen_model, fine_model):
        super().__init__(pbar_log=True, log_level='INFO')
        self.coarsen_model = coarsen_model
        self.fine_model = fine_model


    def refine_and_interpolate(self, k, *functions):
        if not functions:
            return ()
        
        fine_mesh = self.fine_model.mesh



    def run(self):
        self.coarsen_model.run()
        self.fine_model.run()

    def coarsen(self):
        self.coarsen_model.run()