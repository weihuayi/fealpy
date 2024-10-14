import numpy as np


class MassIntegrator:
    """
    @note (u, v)
    """    

    def __init__(self, q=3):
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None):
        pass
