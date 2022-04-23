#!/usr/bin/env python3
# 

import argparse
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import StructureIntervalMesh
from fealpy.timeintegratoralg import UniformTimeLine 

