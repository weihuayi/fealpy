import numpy as np
import matplotlib.pyplot as plt

from gmsher import RB_IGCT_gmshing

domain = RB_IGCT_gmshing()
mesh = domain.meshing()
