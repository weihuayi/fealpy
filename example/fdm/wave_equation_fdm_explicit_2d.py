from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_2d  import MembraneOscillationPDEData
from fealpy.fdm.wave_operator import WaveOperator

pde = MembraneOscillationPDEData()
