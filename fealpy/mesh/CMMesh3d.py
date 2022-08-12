
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse.csgraph import minimum_spanning_tree

from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .Mesh2d import Mesh2d
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from ..common import DynamicArray

class CombinMapMesh3d():
    def __init__(self, dart, node):
        """!
        @param dart : (v, e, f, c, b1, b2, b3)
        """
        self.itype = dart.dtype
        self.ftype = node.dtype

        self.node = DynamicArray(node, dtype = node.dtype)
        self.ds = CombinMapMeshDataStructure(dart, NN = node.shape[0])
        self.meshtype = 'dart3d'

        self.dartdata = {}
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.meshdata = {}


class CombinMapMeshDataStructure():
    def __init__(self, dart):
        self.dart = dart
        ND = dart.shape[0]
        NN, NE, NF, NC = np.max(dart[:, :4])+1

        self.hnode = np.zeros(NN, dtype=np.int_)
        self.hedge = np.zeros(NE, dtype=np.int_)
        self.hface = np.zeros(NF, dtype=np.int_)
        self.hcell = np.zeros(NC, dtype=np.int_)

        self.hnode[dart[:, 0]] = np.arange(ND) 
        self.hedge[dart[:, 1]] = np.arange(ND) 
        self.hface[dart[:, 2]] = np.arange(ND) 
        self.hcell[dart[:, 3]] = np.arange(ND) 

    def cell_to_face(self):
        cf = self.dart[:, [3, 2]]
        NC = len(self.hcell)

        cell2face = np.unique(cf, axis=0)
        cell2faceLocation = np.zeros(NC+1, dtype=np.int_)
        np.add.at(cell2faceLocation[1:], cell2face[:, 0], 1)

        cell2faceLocation[:] = np.cumsum(cell2faceLocation)
        return cell2face[:, 1], cell2faceLocation

    def cell_to_edge(self):
        ce = self.dart[:, [3, 1]]
        NC = len(self.hcell)

        cell2edge = np.unique(ce, axis=0)
        cell2edgeLocation = np.zeros(NC+1, dtype=np.int_)
        np.add.at(cell2edgeLocation[1:], cell2edge[:, 0], 1)

        cell2edgeLocation[:] = np.cumsum(cell2edgeLocation)
        return cell2edge[:, 1], cell2edgeLocation

    def cell_to_node(self):
        cv = self.dart[:, [3, 0]]
        NC = len(self.hcell)

        cell2node = np.unique(cv, axis=0)
        cell2nodeLocation = np.zeros(NC+1, dtype=np.int_)
        np.add.at(cell2nodeLocation[1:], cell2node[:, 0], 1)

        cell2nodeLocation[:] = np.cumsum(cell2nodeLocation)
        return cell2node[:, 1], cell2nodeLocation
















