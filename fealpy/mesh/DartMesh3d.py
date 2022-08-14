
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse.csgraph import minimum_spanning_tree

from ..quadrature import TriangleQuadrature, QuadrangleQuadrature, GaussLegendreQuadrature 
from .Mesh2d import Mesh2d
from .adaptive_tools import mark
from .mesh_tools import show_halfedge_mesh
from ..common import DynamicArray

class DartMesh3d():
    def __init__(self, node, dart):
        """!
        @param dart : (v, e, f, c, b1, b2, b3)
        """
        self.itype = dart.dtype
        self.ftype = node.dtype

        self.node = node
        self.ds = DartMeshDataStructure(dart)
        self.meshtype = 'dart3d'

        self.dartdata = {}
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.meshdata = {}

    @classmethod
    def from_mesh(cls, mesh):
        """!
        @brief 输入一个四面体或六面体网格，将其转化为 dart 数据结构。
        """
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        locF2E = mesh.ds.localFace2edge
        locEdge = mesh.ds.localEdge
        cell2edge = mesh.ds.cell_to_edge()
        cell2face = mesh.ds.cell_to_face()
        face2edge = mesh.ds.face_to_edge()
        face2cell = mesh.ds.face_to_cell()

        NEC = mesh.ds.NEC
        NFC = mesh.ds.NFC
        NVF = mesh.ds.NVF
        locE2F = -np.ones([NEC, 2], dtype=np.int_)
        for i in range(NFC): 
            e = locF2E[i]
            flag = locE2F[e, 0] < 0
            locE2F[e[flag], 0] = i
            locE2F[e[~flag], 1] = i

        ND = 2*NEC*NC
        dart = -np.ones([ND, 7], dtype=np.int_)

        ## 顶点
        dart[::2, 0] = cell[:, locEdge[:, 1]].flat
        dart[1::2, 0] = cell[:, locEdge[:, 0]].flat

        ## 边
        dart[:, 1] = np.repeat(cell2edge.flat, 2)

        ## 面
        dart[::2, 2] = cell2face[:, locE2F[:, 0]].flat
        dart[1::2, 2] = cell2face[:, locE2F[:, 1]].flat

        ## 单元
        dart[:, 3] = np.repeat(np.arange(NC), 2*NEC)

        ## b2
        dart[::2, 5] = np.arange(ND)[1::2]
        dart[1::2, 5] = np.arange(ND)[::2]

        ## b1, b3
        face2dart = -np.ones([NF, 2, NVF], dtype=np.int_)
        nex = np.roll(np.arange(NVF), NVF-1)
        for i in range(NFC):
            f = cell2face[:, i]
            lf2e = cell2edge[:, locF2E[i]] # 局部的 face2edge
            gf2e = face2edge[f] # 全局的 face2edge

            flag = (locE2F[locF2E[i], 0]!=i).astype(np.int_).reshape(1, -1)
            lf2dart = locF2E[i]*2+np.arange(NC)[:, None]*NEC*2+flag
            dart[lf2dart, 4] = lf2dart[:, nex]

            idx = np.argsort(lf2e, axis=1)
            idx1 = np.argsort(np.argsort(gf2e, axis=1), axis=1)
            idx = idx[np.arange(NC)[:, None], idx1]

            ## 此时有 gf2e = lf2e[np.arange(NC)[:, None], idx]
            flag = face2cell[f, 0]==np.arange(NC)
            if np.any(flag):
                face2dart[f[flag], 0] = lf2dart[np.arange(NC)[flag, None], idx[flag]]
            if np.any(~flag):
                face2dart[f[~flag], 1] = lf2dart[np.arange(NC)[~flag, None], idx[~flag]]

        flag = np.any(face2dart[:, 1]<0, axis=1)
        face2dart[flag, 1] = face2dart[flag, 0]

        dart[face2dart[:, 0], -1] = face2dart[:, 1]
        dart[face2dart[:, 1], -1] = face2dart[:, 0]
        
        mesh =  cls(node, dart)
        return mesh

    def number_of_nodes(self):
        return len(self.node)

    def number_of_edges(self):
        return len(self.ds.hedge)

    def number_of_faces(self):
        return len(self.ds.hface)

    def number_of_cells(self):
        return len(self.ds.hcell)

    def to_vtk(self):
        NC = self.number_of_cells()

        node = self.node
        cell, cellLoc = self.ds.cell_to_node()
        face, faceLoc = self.ds.face_to_node()
        cell2face, cell2faceLoc = self.ds.cell_to_face()

        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        uGrid = vtkUnstructuredGrid()
        uGrid.SetPoints(points)
        for i in range(NC):
            FacesIdList = vtkIdList()

            F = cell2faceLoc[i+1]-cell2faceLoc[i]
            FacesIdList.InsertNextId(F)

            for j in range(F):
                f = face[faceLoc[cell2face[i, j]]:faceLoc[cell2face[i, j]+1]]
                FacesIdList.InsertNextId(len(f))
                [FacesIdList.InsertNextId(k) for k in f]

            uGrid.InsertNextCell(VTK_POLYHEDRON, FacesIdList)

        pdata = mesh.GetPointData()
        if nodedata is not None:
            for key, val in nodedata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])
                    d.SetName(key)
                    pdata.AddArray(d)

        if celldata is not None:
            cdata = mesh.GetCellData()
            for key, val in celldata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = np.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == np.bool:
                        d = vnp.numpy_to_vtk(val1.astype(np.int_))
                    else:
                        d = vnp.numpy_to_vtk(val1[:])

                    d.SetName(key)
                    cdata.AddArray(d)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(uGrid)
        writer.Write()

class DartMeshDataStructure():
    def __init__(self, dart):
        """!
        @brief hcell, hface, hedge 给出了 cell, face, edge 的定向
        """
        self.dart = dart
        ND = dart.shape[0]
        NN, NE, NF, NC = np.max(dart[:, :4], axis=0)+1

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

    def cell_to_cell(self):
        dart = self.dart
        cf = dart[:, [3, 2]]
        NC = len(self.hcell)

        cell2face, dartIdx = np.unique(cf, axis=0, return_index=True)
        cell2cell = dart[dart[dartIdx, 6], 3] 

        cell2cellLocation = np.zeros(NC+1, dtype=np.int_)
        np.add.at(cell2cellLocation[1:], cell2face[:, 0], 1)

        cell2cellLocation[:] = np.cumsum(cell2cellLocation)
        return cell2cell, cell2cellLocation

    def face_to_edge(self):
        NF = len(self.hface)
        dart = self.dart

        # get the number of edge of face
        NEF = np.zeros(NF, dtype=np.int_) 
        fe = dart[:, [2, 1]]
        f2e = np.unique(fe, axis=0) #没有定向的 face2edge
        np.add.at(NEF, f2e[:, 0], 1)

        f2eLocation = np.zeros(NF+1, dtype=self.itype)
        f2eLocation[1:] = np.cumsum(NEF)

        f2e[:] = 0 
        current = self.hface.copy() #循环的dart
        idx = f2eLocation[:-1].copy() #循环的 f2e 索引
        isNotOK = np.ones(NF, dtype=np.bool_)
        while np.any(isNotOK):
            f2e[idx[isNotOK]] = dart[current[isNotOK], 1]
            current[isNotOK] = dart[current[isNotOK], 4]
            idx[isNotOK] += 1
            isNotOK = idx < f2eLocation[1:]
        return f2e, f2eLocation

    def face_to_node(self):
        """!
        @brief 获得每个面的顶点，顶点按照面的定向逆时针排列.
        """
        NF = len(self.hface)
        dart = self.dart

        # get the number of vertex of face
        NVF = np.zeros(NF, dtype=np.int_) 
        fn = dart[:, [2, 0]]
        f2n = np.unique(fn, axis=0) #没有定向的 face2node
        np.add.at(NVF, f2n[:, 0], 1)

        f2nLocation = np.zeros(NF+1, dtype=self.itype)
        f2nLocation[1:] = np.cumsum(NVF)

        current = self.hface.copy() #循环的dart
        idx = f2nLocation[:-1].copy() #循环的 f2n 索引
        isNotOK = np.ones(NF, dtype=np.bool_)
        while np.any(isNotOK):
            f2n[idx[isNotOK]] = dart[current[isNotOK], 1]
            current[isNotOK] = dart[current[isNotOK], 4]
            idx[isNotOK] += 1
            isNotOK = idx < f2nLocation[1:]
        return f2n, f2nLocation

    def face_to_cell(self):
        """!
        @brief 获取面相邻的单元的编号 [c0, c1, c0中局部编号, c1中局部编号]
        """
        dart = self.dart
        hface = self.hface
        NF = len(hface)

        f2c = -np.ones([NF, 4], dtype=np.int_)
        f2c[hface, 0] = dart[hface, 3]
        f2c[hface, 1] = dart[dart[hface, 6], 3]

        cell2face, cell2faceLoc = self.cell_to_face()

        i = 0
        c = np.arange(NC)
        isNotOK = np.ones(NC, dtype=np.bool_)
        idx = cell2faceLoc[:-1].copy()
        while np.any(isNotOK):
            f = cell2face[idx[isNotOK]]
            flag = f2c[f, 0]==c[isNotOK]
            f2c[f[flag], 2] = i
            f2c[f[~flag], 3] = i

            idx[isNotOK] = idx[isNotOK]+1
            isNotOK = idx < cell2faceLoc[1:]
            i += 1

        flag = f2c[:, 3]<0
        f2c[flag, 3] = f2c[flag, 2]
        return f2c

    def edge_to_node(self):
        dart = self.dart
        hedge = self.hedge
        NE = len(hedge)

        e2n = np.zeros([NE, 2], dtype=np.int_)
        e2n[hedge, 1] = dart[hedge, 0]
        e2n[hedge, 0] = dart[dart[hedge, 5], 0]
        return e2n

    def edge_to_face(self):
        pass

    def edge_to_cell(self):
        pass

    def node_to_node(self):
        pass

    def node_to_edge(self):
        pass

    def node_to_face(self):
        pass

    def node_to_cell(self):
        pass


