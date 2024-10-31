import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fealpy.pde.lcy  import CahnHilliardData5
from fealpy.fem.CahnHilliardRFEMModel import CahnHilliardRFEMModel 



pde = CahnHilliardData5(0, 0.0001)
fem = CahnHilliardRFEMModel(pde, 8, 0.00001, 3)
fem.solve()




