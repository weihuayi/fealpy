import sys
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0),
    (3, 0, 2)], dtype=np.int)

tmesh = TriangleMesh(node, cell)
tmesh.uniform_refine()
nodeIMatrix, cellIMatrix = tmesh.uniform_refine(returnim=True)
nodeIMatrix = nodeIMatrix[0]
print('nodeIMatrix', nodeIMatrix.toarray())
