import numpy as np


class TrussStructureIntegrator:
    def __init__(self, E, A):
        self.E = E 
        self.A = A 


    def assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):

        c = self.E*self.A
        mesh = space0.mesh
        GD = mesh.geo_dimension()
        NC = mesh.nubmer_of_cells()
        l = mesh.entity_measure('cell')
        tan = mesh.cell_unit_tangent()

        R = np.einsum('ik, im->ikm', tan, tan)
        K = np.zeros((NC, 2*GD, 2*GD), dtype=np.float64)
        K[:, :GD, :GD] = R
        K[:, -GD:, :GD] = -R
        K[:, :GD, -GD:] = -R
        K[:, -GD:, -GD:] = R
        K *= c 
        K /= l[:, None]

        return K
