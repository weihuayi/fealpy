import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fealpy.pde.lcy  import CahnHilliardData5
from fealpy.fem.CahnHilliardRFEMModel import CahnHilliardRFEMModel 


def init():
    return lines, points

#def animate(i):
#    global fem
#    fem.step()
   

pde = CahnHilliardData5(0, 0.0001)
fem = CahnHilliardRFEMModel(pde, 8, 0.00001, 3)
fem.solve()




