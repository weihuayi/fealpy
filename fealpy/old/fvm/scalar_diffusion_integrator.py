#!/usr/bin/python3
import numpy as np
from scipy.sparse import csr_matrix

class ScalarDiffusionIntegrator:
    def __init__(self, mesh, c=None):
        self.c = c
        self.mesh = mesh

    def cell_center_matrix(self, bf, is_bf_boundary):
        c = self.c
        mesh = self.mesh

        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell2edge = mesh.ds.cell_to_edge()

        import ipdb
        h = mesh.entity_measure('edge')[0]

        I = np.arange(NC)
        J = np.arange(NC)
        val = 4*h/h*np.ones(NC)
        A = csr_matrix((val,(I, J)), shape=(NC, NC))


        c2c = mesh.ds.cell_to_cell()
        flag = c2c[np.arange(NC)] == np.arange(NC)[:, None] # 判断是边界的

        I = np.where(flag)[0] 
        J = np.where(flag)[0]
        val = (h/(h/2)-h/h)*np.ones(I.shape)
        A += csr_matrix((val,(I, J)), shape=(NC, NC))


        I = np.where(~flag)[0] 
        J = c2c[~flag]
        val = -h/h*np.ones(I.shape)
        A += csr_matrix((val,(I, J)), shape=(NC, NC))

        b = np.zeros(NC)
        index = np.where(flag)[0]
        index1 = cell2edge[flag]
        point = node[edge[index1]]
        point = point[:,0,:]*(1/2)+point[:,1,:]*(1/2)
        flag = is_bf_boundary(point)
        bu = bf(point)[flag]
        data = bu*h/(h/2)
        np.add.at(b, index, data)

        return A,b

    def node_center_matrix(self):
        pass
