import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

from fealpy.femmodel.CahnHilliardRFEMModel import CahnHilliardRFEMModel 
from fealpy.model.lcy import CahnHilliardData1

from fealpy.quadrature.TriangleQuadrature import TriangleQuadrature 

pde = CahnHilliardData1(0, 1, alpha=0.125)
integrator = TriangleQuadrature(3)
fem = CahnHilliardRFEMModel(pde,4,0.0001,integrator)
fem.solve()

#print(fem.uh)
