from typing import Union, Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .plot import Plotable
from .mesh_base import Mesh
from . import TetrahedronMesh, HexahedronMesh

class DartMesh(Mesh, Plotable):
    """Dart mesh class.

    Parameters
    ----------
    node : TensorLike
        Node coordinates of the mesh.
    dart : TensorLike
        A dart is represented as a tuple (v, e, f, c) consisting of the dart's vertex v,
        its associated edge e, face f, and cell c.
        Additionally, three mappings are included: next edge b1, opposite edge in the same cell b2,
        and opposite edge in a different cell b3.
        Therefore, the overall dart data structure is (v, e, f, c, b1, b2, b3).
    """

    def __init__(self, node: TensorLike, dart: TensorLike):
        super().__init__(TD=3, itype=dart.dtype, ftype=node.dtype)

        self.node = node
        self.dart = dart

        # Initialize the data structure for the dart mesh
        ND = dart.shape[0]
        NN, NE, NF, NC = bm.max(dart[:, :4], axis=0) + 1

        self.hnode = bm.zeros(NN, dtype=bm.int64)
        self.hedge = bm.zeros(NE, dtype=bm.int64)
        self.hface = bm.zeros(NF, dtype=bm.int64)
        # 每一个 cell（单元）在 dart 表中的起始索引或代表 dart 的编号集合
        self.hcell = bm.zeros(NC, dtype=bm.int64)

        self.hnode = bm.set_at(self.hnode, dart[:, 0], bm.arange(ND))
        self.hedge = bm.set_at(self.hedge, dart[:, 1], bm.arange(ND))
        self.hface = bm.set_at(self.hface, dart[:, 2], bm.arange(ND))
        self.hcell = bm.set_at(self.hcell, dart[:, 3], bm.arange(ND))
        # ==============================================

        self.meshtype = 'dart3d'

        self.construct()
        # ==============================================

        self.dartdata = {}
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.meshdata = {}

    def construct(self):
        """
        Construct the dart mesh by creating mappings between different entities.
        Returns
        -------

        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        self.edge = self.edge_to_node()
        self.face = self.face_to_node()
        self.cell = self.cell_to_node()

        self.face2edge = self.face_to_edge()
        self.face2cell = self.face_to_cell()
        self.cell2edge = self.cell_to_edge()
        self.cell2face = self.cell_to_face()



    def number_of_darts(self):
        return len(self.dart)

    def number_of_nodes(self):
        return len(self.node)

    def number_of_edges(self):
        return len(self.hedge)

    def number_of_faces(self):
        return len(self.hface)

    def number_of_cells(self):
        return len(self.hcell)

    def cell_to_face(self, index: Index = _S):
        cf = self.dart[:, [3, 2]]
        NC = len(self.hcell)

        cell2face = bm.unique(cf, axis=0)
        cell2faceLocation = bm.zeros(NC + 1, dtype=bm.int64)
        cell2faceLocation = bm.set_at(cell2faceLocation, slice(1, None),
                                      bm.index_add(cell2faceLocation[1:], cell2face[:, 0], 1))
        cell2faceLocation = bm.cumsum(cell2faceLocation, axis=0)
        return cell2face[:, 1], cell2faceLocation

    def cell_to_edge(self):
        ce = self.dart[:, [3, 1]]
        NC = len(self.hcell)

        cell2edge = bm.unique(ce, axis=0)
        cell2edgeLocation = bm.zeros(NC + 1, dtype=bm.int64)
        cell2edgeLocation = bm.set_at(cell2edgeLocation, slice(1, None),
                                      bm.index_add(cell2edgeLocation[1:], cell2edge[:, 0], 1))
        cell2edgeLocation = bm.cumsum(cell2edgeLocation, axis=0)
        return cell2edge[:, 1], cell2edgeLocation

    def cell_to_node(self):
        cv = self.dart[:, [3, 0]]
        NC = len(self.hcell)

        cell2node = bm.unique(cv, axis=0)
        cell2nodeLocation = bm.zeros(NC + 1, dtype=bm.int64)
        cell2nodeLocation = bm.set_at(cell2nodeLocation, slice(1, None),
                                      bm.index_add(cell2nodeLocation[1:], cell2node[:, 0], 1))
        cell2nodeLocation = bm.cumsum(cell2nodeLocation, axis=0)
        return cell2node[:, 1], cell2nodeLocation

    def cell_to_cell(self):
        dart = self.dart
        cf = dart[:, [3, 2]]
        NC = len(self.hcell)

        cell2face, dartIdx = bm.unique(cf, axis=0, return_index=True)
        cell2cell = dart[dart[dartIdx, 6], 3]

        cell2cellLocation = bm.zeros(NC + 1, dtype=bm.int64)
        cell2cellLocation = bm.set_at(cell2cellLocation, slice(1, None),
                                      bm.index_add(cell2cellLocation[1:], cell2face[:, 0], 1))
        cell2cellLocation = bm.cumsum(cell2cellLocation, axis=0)
        return cell2cell, cell2cellLocation

    def face_to_edge(self):
        NF = len(self.hface)
        dart = self.dart

        NEF = bm.zeros(NF, dtype=bm.int64)
        fe = dart[:, [2, 1]]
        f2e = bm.unique(fe, axis=0)
        NEF = bm.index_add(NEF, f2e[:, 0], 1)

        f2eIndices = bm.cumsum(bm.concat([bm.zeros(1, dtype=bm.int64), NEF]), axis=0)
        f2e = bm.zeros(f2eIndices[-1], dtype=bm.int64)

        current = bm.copy(self.hface)
        idx = bm.copy(f2eIndices[:-1])
        isNotOK = bm.ones(NF, dtype=bool)

        while bm.any(isNotOK):
            f2e = bm.set_at(f2e, idx[isNotOK], dart[current[isNotOK], 1])
            current = bm.set_at(current, isNotOK, dart[current[isNotOK], 4])
            idx = bm.set_at(idx, isNotOK, idx[isNotOK] + 1)
            isNotOK = idx < f2eIndices[1:]

        return f2e, f2eIndices

    def face_to_node(self, index: Index = _S):
        NF = len(self.hface)
        dart = self.dart

        NVF = bm.zeros(NF, dtype=bm.int64)
        fn = dart[:, [2, 0]]
        f2n = bm.unique(fn, axis=0)
        NVF = bm.index_add(NVF, f2n[:, 0], 1)

        N = len(self.hface[index])
        f2nLocation = bm.zeros(N + 1, dtype=bm.int64)
        f2nLocation = bm.set_at(f2nLocation, slice(1, None), bm.cumsum(NVF[index], axis=0))

        f2n = bm.zeros(f2nLocation[-1], dtype=bm.int64)

        current = bm.copy(self.hface[index])
        idx = bm.copy(f2nLocation[:-1])
        isNotOK = bm.ones(N, dtype=bool)

        while bm.any(isNotOK):
            f2n = bm.set_at(f2n, idx[isNotOK], dart[current[isNotOK], 0])
            current = bm.set_at(current, isNotOK, dart[current[isNotOK], 4])
            idx = bm.set_at(idx, isNotOK, idx[isNotOK] + 1)
            isNotOK = idx < f2nLocation[1:]

        return f2n, f2nLocation

    def face_to_cell(self):
        dart = self.dart
        hface = self.hface
        NF = len(hface)
        NC = len(self.hcell)

        f2c = -bm.ones([NF, 4], dtype=bm.int64)
        f2c = bm.set_at(f2c, (slice(None), 0), dart[hface, 3])
        f2c = bm.set_at(f2c, (slice(None), 1), dart[dart[hface, 6], 3])

        cell2face, cell2faceLoc = self.cell_to_face()
        idx = bm.copy(cell2faceLoc[:-1])
        i = 0
        c = bm.arange(NC)
        isNotOK = bm.ones(NC, dtype=bool)

        while bm.any(isNotOK):
            f = cell2face[idx[isNotOK]]
            flag = f2c[f, 0] == c[isNotOK]
            f2c = bm.set_at(f2c, (f[flag], 2), i)
            f2c = bm.set_at(f2c, (f[~flag], 3), i)
            idx = bm.set_at(idx, isNotOK, idx[isNotOK] + 1)
            isNotOK = idx < cell2faceLoc[1:]
            i += 1

        flag = f2c[:, 3] < 0
        f2c = bm.set_at(f2c, (flag, 3), f2c[flag, 2])
        return f2c

    def edge_to_node(self, index: Index = _S):
        dart = self.dart
        hedge = self.hedge[index]
        NE = len(hedge)

        e2n = bm.zeros((NE, 2), dtype=bm.int64)
        e2n = bm.set_at(e2n, (slice(None), 1), dart[hedge, 0])
        e2n = bm.set_at(e2n, (slice(None), 0), dart[dart[hedge, 5], 0])
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

    def boundary_dart_flag(self):
        ND = len(self.dart)
        dart = self.dart
        isBDDart = dart[:, -1] == bm.arange(ND)
        return isBDDart

    def boundary_node_flag(self):
        NN = len(self.hnode)
        isBDNode = bm.zeros(NN, dtype=bm.bool)

        isBDDart = self.boundary_dart_flag()
        indices = self.dart[isBDDart, 0]
        isBDNode = bm.set_at(isBDNode, indices, True)
        return isBDNode

    def boundary_edge_flag(self):
        NE = len(self.hedge)
        isBDEdge = bm.zeros(NE, dtype=bm.bool)

        isBDDart = self.boundary_dart_flag()
        indices = self.dart[isBDDart, 1]
        isBDEdge = bm.set_at(isBDEdge, indices, True)
        return isBDEdge

    def boundary_face_flag(self):
        NF = len(self.hface)
        isBDFace = bm.zeros(NF, dtype=bm.bool)

        isBDDart = self.boundary_dart_flag()
        indices = self.dart[isBDDart, 2]
        isBDFace = bm.set_at(isBDFace, indices, True)
        return isBDFace

    def boundary_cell_flag(self):
        NC = len(self.hcell)
        isBDCell = bm.zeros(NC, dtype=bm.bool)

        isBDDart = self.boundary_dart_flag()
        indices = self.dart[isBDDart, 3]
        isBDCell = bm.set_at(isBDCell, indices, True)
        return isBDCell

    def boundary_dart_index(self):
        return bm.where(self.boundary_dart_flag())[0]

    def boundary_node_index(self):
        return bm.where(self.boundary_node_flag())[0]

    def boundary_edge_index(self):
        return bm.where(self.boundary_edge_flag())[0]

    def boundary_face_index(self):
        return bm.where(self.boundary_face_flag())[0]

    def boundary_cell_index(self):
        return bm.where(self.boundary_cell_flag())[0]

    def to_vtk(self, fname: str):
        """
        Export the DartMesh to a VTK file.

        Parameters
        ----------
        fname : str
            The name of the file to save the mesh in VTK format.
        """
        try:
            import vtk
            import vtk.util.numpy_support as vnp
        except ImportError:
            raise ImportError("VTK is not installed. Please install it to use this method.")

        NC = self.number_of_cells()

        node = bm.to_numpy(self.node)
        cell, cellLoc = self.cell_to_node()
        face, faceLoc = self.face_to_node()
        cell2face, cell2faceLoc = self.cell_to_face()

        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        uGrid = vtk.vtkUnstructuredGrid()
        uGrid.SetPoints(points)
        for i in range(NC):
            FacesIdList = vtk.vtkIdList()

            F = cell2faceLoc[i+1]-cell2faceLoc[i]
            FacesIdList.InsertNextId(F)

            fl = cell2faceLoc[i]
            for j in range(F):
                f = face[faceLoc[cell2face[fl+j]]:faceLoc[cell2face[fl+j]+1]]
                FacesIdList.InsertNextId(len(f))
                [FacesIdList.InsertNextId(k) for k in f]

            uGrid.InsertNextCell(vtk.VTK_POLYHEDRON, FacesIdList)

        pdata = uGrid.GetPointData()
        # TODO: 是否需要补充其他实体的数据
        if self.nodedata:
            nodedata = self.nodedata
            for key, val in nodedata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = bm.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == bm.bool:
                        d = vnp.numpy_to_vtk(bm.to_numpy(bm.astype(val1, bm.int64)))
                    else:
                        d = vnp.numpy_to_vtk(bm.to_numpy(val1[:]))
                    d.SetName(key)
                    pdata.AddArray(d)

        if self.celldata:
            celldata = self.celldata
            cdata = uGrid.GetCellData()
            for key, val in celldata.items():
                if val is not None:
                    if len(val.shape) == 2 and val.shape[1] == 2:
                        shape = (val.shape[0], 3)
                        val1 = bm.zeros(shape, dtype=val.dtype)
                        val1[:, 0:2] = val
                    else:
                        val1 = val

                    if val1.dtype == bm.bool:
                        d = vnp.numpy_to_vtk(bm.to_numpy(bm.astype(val1, bm.int64)))
                    else:
                        d = vnp.numpy_to_vtk(bm.to_numpy(val1[:]))

                    d.SetName(key)
                    cdata.AddArray(d)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(uGrid)
        writer.Write()

    def print(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        print("hcell:")
        for i, val in enumerate(self.hcell):
            print(i, ':', val)

        print("hedge:")
        for i, val in enumerate(self.hedge):
            print(i, ":", val)

        print("dart:")
        for i, val in enumerate(self.dart):
            print(i, ":", val)

        print("edge:")
        edge = self.edge_to_node()
        for i, val in enumerate(edge):
            print(i, ":", val)

        print("face:")
        face, faceLoc = self.face_to_node()
        for i in range(NF):
            print(i, ":", face[faceLoc[i]:faceLoc[i+1]])

        print("face2edge:")
        face2edge, face2edgeLoc = self.face_to_edge()
        for i in range(NF):
            print(i, ":", face2edge[face2edgeLoc[i]:face2edgeLoc[i+1]])

        print("face2cell:")
        face2cell = self.face_to_cell()
        for i, val in enumerate(face2cell):
            print(i, ":", val)

        print("cell:")
        cell, cellLoc = self.cell_to_node()
        for i in range(NC):
            print(i, ":", cell[cellLoc[i]:cellLoc[i+1]])

        print("cell2edge:")
        cell2edge, cell2edgeLoc = self.cell_to_edge()
        for i in range(NC):
            print(i, ":", cell2edge[cell2edgeLoc[i]:cell2edgeLoc[i+1]])

        print("cell2face:")
        cell2face, cell2faceLoc = self.cell_to_face()
        for i in range(NC):
            print(i, ":", cell2face[cell2faceLoc[i]:cell2faceLoc[i+1]])

    def __str__(self):
        from io import StringIO
        import sys

        buffer = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer
        try:
            self.print()
        finally:
            sys.stdout = sys_stdout
        return buffer.getvalue()

    def dual_mesh(self, dual_point='barycenter')->'Optional[DartMesh]':
        """
        Create a dual mesh from the DartMesh.

        Parameters
        ----------
        dual_point : str, optional
            The type of point to use for the dual mesh. 'barycenter' or 'circumcenter'.

        Returns
        -------
        Optional[DartMesh]
            A new DartMesh representing the dual mesh.
        """
        dart = self.dart
        node = self.node
        bdart = self.boundary_dart_index()
        bnode = self.boundary_node_index()
        bedge = self.boundary_edge_index()
        bface = self.boundary_face_index()

        ND = len(dart)
        NBD = len(bdart)
        NBN = len(bnode)
        NBE = len(bedge)
        NBF = len(bface)
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        # 生成对偶网格的节点
        newnode = bm.zeros((NC + NBN + NBE + NBF, 3), dtype=bm.float64)
        newnode = bm.set_at(newnode, slice(NC, NC+NBN), node[bnode])
        if dual_point == 'barycenter':
            newnode = bm.set_at(newnode, slice(0, NC), self.entity_barycenter('cell'))
            newnode = bm.set_at(newnode, slice(NC + NBN, NC + NBN + NBE),
                      self.entity_barycenter('edge', index=bedge))
            newnode = bm.set_at(newnode, slice(-NBF, None),
                      self.entity_barycenter('face', index=bface))
        elif dual_point == 'circumcenter':
            newnode = bm.set_at(newnode, slice(0, NC), self.entity_circumcenter('cell'))
            newnode = bm.set_at(newnode, slice(NC + NBN, NC + NBN + NBE),
                      self.entity_circumcenter('edge', index=bedge))
            newnode = bm.set_at(newnode, bm.arange(-NBF, None),
                      self.entity_circumcenter('face', index=bface))

        # 边界实体与新节点的映射
        n2n = -bm.ones(NN, dtype=bm.int64)
        e2n = -bm.ones(NE, dtype=bm.int64)
        f2n = -bm.ones(NF, dtype=bm.int64)
        n2n = bm.set_at(n2n, bnode, bm.arange(NC, NC + NBN))
        e2n = bm.set_at(e2n, bedge, bm.arange(NC + NBN, NC + NBN + NBE))
        f2n = bm.set_at(f2n, bface, bm.arange(NC + NBN + NBE, NC + NBN + NBE + NBF))

        # bdart 映射表
        m = int(bm.max(bdart)) + 1
        bdIdxmap = bm.zeros(m, dtype=bm.int64)
        bdIdxmap = bm.set_at(bdIdxmap, bdart, bm.arange(NBD))

        bdart2nn = n2n[dart[bdart, 0]]
        bdart2en = e2n[dart[bdart, 1]]
        bdart2fn = f2n[dart[bdart, 2]]

        bdart2e = bm.zeros((2, m), dtype=bm.int64)
        bdart2e = bm.set_at(bdart2e, (0, bdart), bm.arange(NF, NF + NBD * 2, 2))
        bdart2e = bm.set_at(bdart2e, (1, bdart), bm.arange(NF + 1, NF + NBD * 2, 2))

        index = bm.argsort(dart[bdart, 1])
        bdopp = bm.zeros(m, dtype=bm.int64)
        bdopp = bm.set_at(bdopp, bdart[index[::2]], bdart[index[1::2]])
        bdopp = bm.set_at(bdopp, bdart[index[1::2]], bdart[index[::2]])

        bnidx = bdIdxmap[dart[bdart, 4]]
        bnoidx = bdIdxmap[bdopp[dart[bdart, 4]]]

        # 构造 newdart
        newdart = bm.zeros((ND + 7 * NBD, 7), dtype=bm.int64)

        # 原始 dart 翻转构建
        newdart = bm.set_at(newdart, slice(0, ND), dart)
        newdart = bm.set_at(newdart, (bdart, 6), bm.arange(ND, ND + NBD))

        newdart = bm.set_at(newdart, (slice(0, ND), slice(0, 4)), bm.flip(newdart[:ND, :4], axis=1))
        newdart = bm.set_at(newdart, (slice(0, ND), 4), newdart[dart[:ND, 5], 6])
        newdart = bm.set_at(newdart, (slice(0, ND), 5), newdart[dart[:ND, 4], 6])

        # 每组 dart 构建
        # Block 0
        newdart = bm.set_at(newdart, (slice(ND, ND+NBD), 0), bdart2fn)
        newdart = bm.set_at(newdart, (slice(ND, ND+NBD), slice(1, 3)), newdart[bdart, 1:3])
        newdart = bm.set_at(newdart, (slice(ND, ND+NBD), 3), dart[dart[bdart, 5], 0])
        newdart = bm.set_at(newdart, (newdart[bdart, 5], 4), bm.arange(ND+NBD*6, ND+NBD*7))
        newdart = bm.set_at(newdart, (newdart[bdart, 5], 5), bdart)
        newdart = bm.set_at(newdart, (slice(ND,ND+NBD), 6), bdart)

        # Block 1
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 0), bdart2en)
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 1), bm.arange(NF, NF + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 2), bm.arange(NE, NE + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 4), bm.arange(ND + NBD * 2, ND + NBD * 3))
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 5), bm.arange(ND + NBD * 5, ND + NBD * 6))
        newdart = bm.set_at(newdart, (slice(ND + NBD, ND + NBD * 2), 6), bm.arange(ND + NBD, ND + NBD * 2))

        # Block 2
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 0), bdart2nn)
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 1), bm.arange(NF + NBD, NF + NBD * 2))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 2), bm.arange(NE, NE + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 4), bm.arange(ND + NBD * 3, ND + NBD * 4))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 2, ND + NBD * 3), 6), bm.arange(ND + NBD * 2, ND + NBD * 3))

        # Block 3
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 0), bdart2en[bnoidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 1), bm.arange(NF + NBD, NF + NBD * 2)[bnoidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 2), bm.arange(NE, NE + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 4), bm.arange(ND + NBD * 4, ND + NBD * 5))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 5),
                            bm.arange(ND + NBD * 2, ND + NBD * 3)[bnoidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 3, ND + NBD * 4), 6), bm.arange(ND + NBD * 3, ND + NBD * 4))

        newdart = bm.set_at(newdart, (newdart[ND + NBD * 3:ND + NBD * 4, 5], 5), bm.arange(ND + NBD * 3, ND + NBD * 4))

        # Block 4
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 0), bdart2fn)
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 1), bm.arange(NF, NF + NBD)[bnidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 2), bm.arange(NE, NE + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 4), bm.arange(ND + NBD, ND + NBD * 2))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 5), bm.arange(ND + NBD * 6, ND + NBD * 7))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 4, ND + NBD * 5), 6), bm.arange(ND + NBD * 4, ND + NBD * 5))

        # Block 5
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 0), bdart2fn)
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 1), bm.arange(NF, NF + NBD))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 2), newdart[bdart, 2])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 4), bdart)
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 5), bm.arange(ND + NBD, ND + NBD * 2))
        newdart = bm.set_at(newdart, (slice(ND + NBD * 5, ND + NBD * 6), 6), newdart[ND:ND + NBD, 4])

        # Block 6
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 0), bdart2en[bnoidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 1), bm.arange(NF, NF + NBD)[bnidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 2), newdart[newdart[bdart, 5], 2])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 3), newdart[bdart, 3])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 4),
                            bm.arange(ND + NBD * 5, ND + NBD * 6)[bnoidx])
        newdart = bm.set_at(newdart, (slice(ND + NBD * 6, ND + NBD * 7), 5), bm.arange(ND + NBD * 4, ND + NBD * 5))

        newdart = bm.set_at(newdart, (newdart[ND + NBD * 5:ND + NBD * 6, 6], 6), bm.arange(ND + NBD * 5, ND + NBD * 6))

        return DartMesh(newnode, newdart)

    def entity_barycenter(self, entityType='cell', index: Index = _S):
        """
                获取实体的重心坐标

                Parameters:
                - entityType: 实体类型，可为 'edge'、'face'、'cell'
                - index: 实体索引切片，默认为所有

                Returns:
                - bary: 每个实体的重心，形状为 (N, 3)
                """
        node = self.node

        if entityType == 'edge':
            edge = self.edge_to_node(index=index)
            bary = bm.mean(node[edge], axis=1)
            return bary

        elif entityType == 'face':
            face, faceLoc = self.face_to_node(index=index)
            NV = (faceLoc[1:] - faceLoc[:-1]).reshape(-1, 1)
            NF = NV.shape[0]

            bary = bm.zeros((NF, 3), dtype=bm.float64)
            isNotOK = bm.ones((NF,), dtype=bm.bool)
            start = bm.copy(faceLoc[:-1])

            while bm.any(isNotOK):
                bary = bm.set_at(
                    bary,
                    isNotOK,
                    bary[isNotOK] + node[face[start[isNotOK]]] / NV[isNotOK]
                )
                start = bm.set_at(start, isNotOK, start[isNotOK] + 1)
                isNotOK = start < faceLoc[1:]
            return bary

        elif entityType == 'cell':
            cell, cellLoc = self.cell_to_node()
            NV = (cellLoc[1:] - cellLoc[:-1]).reshape(-1, 1)
            NC = self.number_of_cells()

            bary = bm.zeros((NC, 3), dtype=bm.float64)
            isNotOK = bm.ones((NC,), dtype=bm.bool)
            start = bm.copy(cellLoc[:-1])

            while bm.any(isNotOK):
                bary = bm.set_at(
                    bary,
                    isNotOK,
                    bary[isNotOK] + node[cell[start[isNotOK]]] / NV[isNotOK]
                )
                start = bm.set_at(start, isNotOK, start[isNotOK] + 1)
                isNotOK = start < cellLoc[1:]
            return bary[index]

    def entity_circumcenter(self, entityType='cell', index: Index = _S):
        """
                计算实体的外接圆/球的球心，仅支持:
                - 三角形面（face）
                - 四面体单元（cell）
                - 边缘中点（edge）

                Parameters:
                - entityType: 'edge', 'face', 'cell'
                - index: 实体索引

                Returns:
                - center: 每个实体的球心坐标，形状为 (N, 3)
                """
        node = self.node

        if entityType == 'edge':
            edge = self.edge_to_node(index=index)
            center = bm.mean(node[edge], axis=1)
            return center

        elif entityType == 'face':
            face, faceLoc = self.face_to_node(index=index)
            assert bm.all(faceLoc[1:] - faceLoc[:-1] == 3), "只支持三角面"

            a = bm.sum((node[face[2::3]] - node[face[1::3]]) ** 2, axis=1).reshape(-1, 1)
            b = bm.sum((node[face[0::3]] - node[face[2::3]]) ** 2, axis=1).reshape(-1, 1)
            c = bm.sum((node[face[1::3]] - node[face[0::3]]) ** 2, axis=1).reshape(-1, 1)

            center = (
                             a * (b + c - a) * node[face[::3]] +
                             b * (c + a - b) * node[face[1::3]] +
                             c * (a + b - c) * node[face[2::3]]
                     ) / (
                             a * (b + c - a) +
                             b * (c + a - b) +
                             c * (a + b - c)
                     )
            return center

        elif entityType == 'cell':
            cell, cellLoc = self.cell_to_node()
            assert bm.all(cellLoc[1:] - cellLoc[:-1] == 4), "只支持四面体单元"

            l = bm.sum(node ** 2, axis=1) / 2
            NC = self.number_of_cells()

            A = bm.zeros((NC, 3, 3), dtype=bm.float64)
            A = bm.set_at(A, (slice(None), slice(None), 0), node[cell[1::4]] - node[cell[::4]])
            A = bm.set_at(A, (slice(None), slice(None), 1), node[cell[2::4]] - node[cell[::4]])
            A = bm.set_at(A, (slice(None), slice(None), 2), node[cell[3::4]] - node[cell[::4]])

            Ainv = bm.linalg.inv(A)
            B = bm.stack([
                l[cell[1::4]] - l[cell[::4]],
                l[cell[2::4]] - l[cell[::4]],
                l[cell[3::4]] - l[cell[::4]]
            ], axis=1)

            center = bm.einsum('cij, ci->cj',Ainv, B)
            return center[index]

    @classmethod
    def from_mesh(cls, mesh:Union[TetrahedronMesh, HexahedronMesh])->'Optional[DartMesh]':
        """
        Create a DartMesh from a TetrahedronMesh or HexahedronMesh.
        Parameters
        ----------
        mesh : Union[TetrahedronMesh, HexahedronMesh]
            The mesh to convert.

        Returns
        -------

        """
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        face = mesh.entity('face')
        cell = mesh.entity('cell')

        locF2E = mesh.localFace2edge
        locE2F = mesh.localEdge2face
        locEdge = mesh.localEdge
        cell2edge = mesh.cell_to_edge()
        cell2face = mesh.cell_to_face()
        face2edge = mesh.face_to_edge()
        face2cell = mesh.face_to_cell()

        NEC = mesh.number_of_edges_of_cells()
        NFC = mesh.number_of_faces_of_cells()
        # TODO: 统一接口
        NVF = mesh.number_of_vertices_of_faces()[0].item()

        ND = 2 * NEC * NC
        dart = bm.full([ND, 7], -1, dtype=int)

        ## 顶点
        dart = bm.set_at(dart, (bm.arange(0, ND, 2), 0), cell[:, locEdge[:, 1]].reshape(-1))
        dart = bm.set_at(dart, (bm.arange(1, ND, 2), 0), cell[:, locEdge[:, 0]].reshape(-1))

        ## 边
        dart = bm.set_at(dart, (bm.arange(ND), 1), bm.repeat(cell2edge.reshape(-1), 2))

        ## 面
        dart = bm.set_at(dart, (bm.arange(0, ND, 2), 2), cell2face[:, locE2F[:, 0]].reshape(-1))
        dart = bm.set_at(dart, (bm.arange(1, ND, 2), 2), cell2face[:, locE2F[:, 1]].reshape(-1))

        ## 单元
        dart = bm.set_at(dart, (bm.arange(ND), 3), bm.repeat(bm.arange(NC), 2 * NEC))

        ## b2
        dart = bm.set_at(dart, (bm.arange(0, ND, 2), 5), bm.arange(1, ND, 2))
        dart = bm.set_at(dart, (bm.arange(1, ND, 2), 5), bm.arange(0, ND, 2))

        ## b1, b3
        face2dart = bm.full([NF, 2, NVF], -1, dtype=int)
        nex = bm.roll(bm.arange(NVF), NVF - 1)

        for i in range(NFC):
            f = cell2face[:, i]
            lf2e = cell2edge[:, locF2E[i]]  # 局部的 face2edge
            gf2e = face2edge[f]


            flag = bm.astype(locE2F[locF2E[i], 0] != i, int).reshape(1, -1)
            lf2dart = locF2E[i] * 2 + bm.arange(NC).reshape(-1, 1) * NEC * 2 + flag
            dart = bm.set_at(dart, (lf2dart.reshape(-1), 4), lf2dart[:, nex].reshape(-1))

            idx = bm.argsort(lf2e, axis=1)
            idx1 = bm.argsort(bm.argsort(gf2e, axis=1), axis=1)
            idx = idx[bm.arange(NC)[:, None], idx1]

            flag_mask = face2cell[f, 0] == bm.arange(NC)
            if bm.any(flag_mask):
                face2dart = bm.set_at(face2dart, (f[flag], 0), lf2dart[bm.arange(NC)[flag, None], idx[flag]])
            if bm.any(~flag_mask):
                face2dart = bm.set_at(face2dart, (f[~flag], 1), lf2dart[bm.arange(NC)[~flag, None], idx[~flag]])

        flag = bm.any(face2dart[:, 1] < 0, axis=1)
        face2dart = bm.set_at(face2dart, (flag, 1), face2dart[flag, 0])

        dart = bm.set_at(dart, (face2dart[:, 0].reshape(-1), 6), face2dart[:, 1].reshape(-1))
        dart = bm.set_at(dart, (face2dart[:, 1].reshape(-1), 6), face2dart[:, 0].reshape(-1))

        mesh = cls(node, dart)
        return mesh
