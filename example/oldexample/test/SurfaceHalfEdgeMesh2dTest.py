#!/usr/bin/env python3
# 
import sys
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.geometry import SphereSurface
from fealpy.writer import MeshWriter

class SurfaceHalfEdgemesh2dTest():

    def __init__(self):
        pass

    def read_surface_mesh(self, fname, plot=False):

        data = sio.loadmat(fname)
        node = np.array(data['node'], dtype=np.float64)
        cell = np.array(data['elem'] - 1, dtype=np.int_)

        mesh = TriangleMesh(node, cell)

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        val = np.ones((NE,), dtype=np.bool_)

        cell2cell = coo_matrix(
                (val, (edge2cell[:, 0], edge2cell[:, 1])),
                shape=(NC, NC))
        cell2cell += coo_matrix(
                (val, (edge2cell[:, 1], edge2cell[:, 0])),
                shape=(NC, NC))
        cell2cell = cell2cell.tocsr()

        mtree = minimum_spanning_tree(cell2cell)
        mtree = mtree + mtree.T

        flag = np.asarray(mtree[edge2cell[:, 0], edge2cell[:, 1]]).reshape(-1).astype(np.int)
        index, = np.nonzero(flag == 0) 

        cedge = edge[index]
        nc = len(cedge)
        val = np.ones(nc, dtype=np.bool_)
        cn2cn = coo_matrix((val, (cedge[:, 0], cedge[:, 1])), shape=(NN, NN))
        cn2cn += coo_matrix((val, (cedge[:, 1], cedge[:, 0])), shape=(NN, NN))
        cn2cn = cn2cn.tocsr()
        ctree = minimum_spanning_tree(cn2cn)
        ctree = ctree + ctree.T
        flag = np.asarray(ctree[cedge[:, 0], cedge[:, 1]]).reshape(-1).astype(np.int)
        flag = flag == 0
        idx0 = index[flag] # 没有在生成树中的边
        idx1 = index[~flag]# 在生成树中的边

        gamma = []
        count = np.zeros(NN, dtype=np.int_)
        for i in idx0:
            isKeepEdge1 = np.ones(len(idx1), dtype=np.bool_)
            while True:
                np.add.at(count, edge[i], 1)
                np.add.at(count, edge[idx1[isKeepEdge1]], 1)
                isDEdge = (count[edge[idx1, 0]] == 1) | (count[edge[idx1, 1]] == 1)
                count[:] = 0
                if np.any(isDEdge):
                    isKeepEdge1 = isKeepEdge1 & (~isDEdge)
                else:
                    break
            loop = np.r_['0', i, idx1[isKeepEdge1]]
            gamma.append(loop)

        writer = MeshWriter(mesh)
        writer.write(fname='test.vtu')
        if 0:
            for i, index in enumerate(gamma):
                writer = MeshWriter(mesh, etype='edge', index=index)
                writer.write(fname='test'+str(i)+'.vtu')



test = SurfaceHalfEdgemesh2dTest()

if sys.argv[1] == 'read':
    fname = sys.argv[2] 
    test.read_surface_mesh(fname, plot=False)
