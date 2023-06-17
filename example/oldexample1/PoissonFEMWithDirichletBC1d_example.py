#!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


from fealpy.pde.poisson_1d import CosData 
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.tools.show import showmultirate


