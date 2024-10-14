import numpy as np
from .function import *

def interpolation(u, V):
    uI = FiniteElementFunction(V)
    ipoints =V.interpolation_points() 
    uI[:] = V.interpolation(u) 
    return uI
