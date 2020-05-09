
import numpy as np

from fealpy.mesh import IntervalMesh

node = np.array([[0], [0.5], [1]], dtype=np.float) # (NN, 1) array
cell = np.array([[0, 1]], dtype=np.int) # (NC, 2) array
