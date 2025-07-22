
from typing import Protocol
from ..backend import bm
from ..decorator import variantmethod
from ..model import ComputationalModel



class TimeStepper:
    def __init__(self, 
                 cmodel: ComputationalModel,
                 dt: float=0.01,
                 *,
                 duration= [0.0, 1.0],
                 method: str='explicit'):
        self.cmodel, self.dt, self.duration = cmodel, dt, duration
