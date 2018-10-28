import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack, bmat, spdiags
from numpy import linalg as LA
from fealpy.fem.integral_alg import IntegralAlg
from fealpy.fdm.DarcyFDMModel import DarcyFDMModel
from scipy.sparse.linalg import cg, inv, dsolve, spsolve

class DarcyForchheimerFDMModel():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh

        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        self.uh = np.zeros(NE, dtype=mesh.ftype)
        self.ph = np.zeros(NC, dtype=mesh.ftype)
        self.uh0 = np.zeros(NE, dtype=mesh.ftype)
        self.ph0 = np.zeros(NC, dtype=mesh.ftype)

        isBDEdge = mesh.ds.boundary_edge_flag()
        isYDEdge = mesh.ds.y_direction_edge_flag()
        isXDEdge = mesh.ds.x_direction_edge_flag()
        bc = mesh.entity_barycenter('edge')
        pc = mesh.entity_barycenter('cell')

        self.uI = np.zeros(NE, dtype=mesh.ftype)
        self.uI[isYDEdge] = pde.velocity_x(bc[isYDEdge])
        self.uI[isXDEdge] = pde.velocity_y(bc[isXDEdge])
        self.pI = pde.pressure(pc)

        self.ph[0] = self.pI[0]


