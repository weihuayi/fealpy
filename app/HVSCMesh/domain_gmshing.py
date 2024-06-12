import numpy as np
import matplotlib.pyplot as plt

from gmsher import RB_IGCT_gmsher, BJT_gmsher

def RB_IGCT_gmshing():
    domain = RB_IGCT_gmsher()
    mesh = domain.meshing()
    return mesh

def BJT_gmshing():
    domain = BJT_gmsher()
    mesh = domain.meshing()
    return mesh

mesh = RB_IGCT_gmshing()
#mesh = BJT_gmshing()
