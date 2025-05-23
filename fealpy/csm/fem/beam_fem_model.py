from ...backend import backend_manager as bm

from ..model import ComputationalModel
from ..model import PDEDataManager

class PoissonFDMModel(ComputationalModel):

    def __init__(self, example: str= 'beam2d'):
        self.pde = PDEDataManager('beam').get_example(example) 
        


    def run(self):
        pass


    def linear_system(self):
        pass


    def solve(self):
        pass