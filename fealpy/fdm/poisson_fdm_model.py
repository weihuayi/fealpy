from ..backend import backend_manager as bm

from ..model import ComputationalModel
from ..model import PDEDataManager

class PoissonFDMModel(ComputationalModel):

    def __init__(self, example: str= 'sinsin', maxit: int=4):
        self.pde = PDEDataManager('poisson').get_example(example) 
        self.maxit = maxit


    def run(self):
        pass


    def linear_system(self):
        pass


    def solve(self):
        pass
