import numpy as np

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
        locE2F = mesh.ds.localEdge2face
        locEdge = mesh.ds.localEdge
        cell2edge = mesh.ds.cell_to_edge()
        cell2face = mesh.ds.cell_to_face()
        face2edge = mesh.ds.face_to_edge()
        face2cell = mesh.ds.face_to_cell()

        NEC = mesh.ds.NEC
        NFC = mesh.ds.NFC
        NVF = mesh.ds.NVF

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

    def dual_mesh(self):
        dart = self.ds.dart
        node = self.node
        bdart = self.ds.boundary_dart_index()
        bnode = self.ds.boundary_node_index()
        bedge = self.ds.boundary_edge_index()
        bface = self.ds.boundary_face_index()
        self.print()
        for i in bdart:
            print(i, ' : ', dart[i])
        
        ND = len(dart)
        NBD = len(bdart)
        NBN = len(bnode)
        NBE = len(bedge)
        NBF = len(bface)
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        print(NN, NE, NF, NC)

        ######## 生成对偶网格的节点##########
        newnode = np.zeros([NC+NBN+NBE+NBF, 3], dtype=np.float_)
        newnode[:NC] = self.entity_barycenter('cell')
        newnode[NC:NC+NBN] = node[bnode]
        newnode[NC+NBN:NC+NBN+NBE] = self.entity_barycenter('edge', index=bedge)
        newnode[-NBF:] = self.entity_barycenter('face', index=bface)
        ####################################

        ############## bdart 与对偶网格边界上的实体间的关系 #################
        # 1. 点边面与新节点的对应关系
        n2n = -np.ones(NN, dtype=np.int_)
        e2n = -np.ones(NE, dtype=np.int_)
        f2n = -np.ones(NF, dtype=np.int_)
        n2n[bnode] = np.arange(NC, NC+NBN) # node 对应的新节点编号
        e2n[bedge] = np.arange(NC+NBN, NC+NBN+NBE)# edge 上的新节点编号
        f2n[bface] = np.arange(NC+NBN+NBE, NC+NBN+NBE+NBF)# face 上的新节点编号

        # 2. 边界 dart 与自身顺序的 map
        m = np.max(bdart)+1
        bdIdxmap = np.zeros(m, dtype=np.int_)
        bdIdxmap[bdart] = np.arange(NBD)

        # 3. 边界 dart 与新节点的对应关系
        bdart2nn = n2n[dart[bdart, 0]] # bdart 对应的节点上的新节点
        bdart2en = e2n[dart[bdart, 1]] # bdart 对应的边上的新节点
        bdart2fn = f2n[dart[bdart, 2]] # bdart 对应的面上的新节点

        # 4. 边界 dart 与新边的对应关系
        bdart2e = np.zeros([2, m], dtype=np.int_)
        bdart2e[:, bdart] = np.arange(NF, NF+NBD*2).reshape(2, -1)

        # 5. 边界 dart 作为边界 halfedge 网格中的 dart 的对边 
        bdopp = np.zeros(m, dtype=np.int_)
        index = np.argsort(dart[bdart, 1])
        bdopp[bdart[index[::2]]] = bdart[index[1::2]]
        bdopp[bdart[index[1::2]]] = bdart[index[::2]]

        bnoidx = bdIdxmap[bdopp[dart[bdart, 4]]] # bdart 下一条边的对边
        ##################################################################

        ############ 生成对偶网格中的 dart #############
        newdart = np.zeros([ND+7*NBD, 7], dtype=np.int_)

        # -1: (v, e, f, c, b1, b2, b3) -> (v, e, f, c, b3(b2), b3(b1), b3)
        newdart[:ND] = dart
        newdart[bdart, 6] = np.arange(ND, ND+NBD) # 给边界 dart 生成 b3

        newdart[:ND, :4] = newdart[:ND, 3::-1]
        newdart[:ND, 4] = newdart[dart[:ND, 5], 6]
        newdart[:ND, 5] = newdart[dart[:ND, 4], 6]

        # 0
        newdart[ND:ND+NBD, 0] = bdart2fn
        newdart[ND:ND+NBD, 1:3] = newdart[bdart, 1:3]
        newdart[ND:ND+NBD, 3] = dart[dart[bdart, 5], 0]
        newdart[newdart[bdart, 5], 4] = np.arange(ND+NBD*6, ND+NBD*7) #TODO
        newdart[newdart[bdart, 5], 5] = bdart # TODO
        newdart[ND:ND+NBD, 6] = bdart

        # 1
        newdart[ND+NBD:ND+NBD*2, 0] = bdart2en
        newdart[ND+NBD:ND+NBD*2, 1] = np.arange(NF, NF+NBD)
        newdart[ND+NBD:ND+NBD*2, 2] = np.arange(NE, NE+NBD)
        newdart[ND+NBD:ND+NBD*2, 3] = newdart[bdart, 3]
        newdart[ND+NBD:ND+NBD*2, 4] = np.arange(ND+NBD*2, ND+NBD*3)
        newdart[ND+NBD:ND+NBD*2, 5] = np.arange(ND+NBD*5, ND+NBD*6)
        newdart[ND+NBD:ND+NBD*2, 6] = np.arange(ND+NBD, ND+NBD*2) 

        # 2 : b2 没有实现，在 3 中实现
        newdart[ND+NBD*2:ND+NBD*3, 0] = bdart2nn
        newdart[ND+NBD*2:ND+NBD*3, 1] = np.arange(NF+NBD, NF+NBD*2)
        newdart[ND+NBD*2:ND+NBD*3, 2] = np.arange(NE, NE+NBD)
        newdart[ND+NBD*2:ND+NBD*3, 3] = newdart[bdart, 3]
        newdart[ND+NBD*2:ND+NBD*3, 4] = np.arange(ND+NBD*3, ND+NBD*4)
        newdart[ND+NBD*2:ND+NBD*3, 6] = np.arange(ND+NBD*2, ND+NBD*3) 

        # 3
        newdart[ND+NBD*3:ND+NBD*4, 0] = bdart2en[bnoidx]
        newdart[ND+NBD*3:ND+NBD*4, 1] = np.arange(NF+NBD, NF+NBD*2)[bnoidx]
        newdart[ND+NBD*3:ND+NBD*4, 2] = np.arange(NE, NE+NBD)
        newdart[ND+NBD*3:ND+NBD*4, 3] = newdart[bdart, 3]
        newdart[ND+NBD*3:ND+NBD*4, 4] = np.arange(ND+NBD*4, ND+NBD*5)
        newdart[ND+NBD*3:ND+NBD*4, 5] = np.arange(ND+NBD*2, ND+NBD*3)[bnoidx]
        newdart[ND+NBD*3:ND+NBD*4, 6] = np.arange(ND+NBD*3, ND+NBD*4) 

        print(bnoidx)
        print('aaa = ', newdart[ND+NBD*3:ND+NBD*4, 5])
        print('bbb = ', np.arange(ND+NBD*2, ND+NBD*3))
        newdart[newdart[ND+NBD*3:ND+NBD*4, 5], 5] = np.arange(ND+NBD*3, ND+NBD*4)

        # 4
        newdart[ND+NBD*4:ND+NBD*5, 0] = bdart2fn
        newdart[ND+NBD*4:ND+NBD*5, 1] = np.arange(NF+NBD, NF+NBD*2)[bnoidx]
        newdart[ND+NBD*4:ND+NBD*5, 2] = np.arange(NE, NE+NBD)
        newdart[ND+NBD*4:ND+NBD*5, 3] = newdart[bdart, 3]
        newdart[ND+NBD*4:ND+NBD*5, 4] = np.arange(ND+NBD, ND+NBD*2)
        newdart[ND+NBD*4:ND+NBD*5, 5] = np.arange(ND+NBD*6, ND+NBD*7)
        newdart[ND+NBD*4:ND+NBD*5, 6] = np.arange(ND+NBD*4, ND+NBD*5) 

        # 5
        print('a = ', np.arange(ND+NBD*5, ND+NBD*6))
        print('b = ', bdart)
        newdart[ND+NBD*5:ND+NBD*6, 0] = bdart2fn
        newdart[ND+NBD*5:ND+NBD*6, 1] = np.arange(NF, NF+NBD)
        newdart[ND+NBD*5:ND+NBD*6, 2] = newdart[bdart, 2]
        newdart[ND+NBD*5:ND+NBD*6, 3] = newdart[bdart, 3]
        newdart[ND+NBD*5:ND+NBD*6, 4] = bdart
        newdart[ND+NBD*5:ND+NBD*6, 5] = np.arange(ND+NBD, ND+NBD*2)
        newdart[ND+NBD*5:ND+NBD*6, 6] = newdart[ND:ND+NBD, 4] 

        # 6
        newdart[ND+NBD*6:ND+NBD*7, 0] = bdart2en[bnoidx]
        newdart[ND+NBD*6:ND+NBD*7, 1] = NF+bdIdxmap[dart[bdart, 4]]
        newdart[ND+NBD*6:ND+NBD*7, 2] = newdart[newdart[bdart, 5], 2]
        newdart[ND+NBD*6:ND+NBD*7, 3] = newdart[bdart, 3]
        newdart[ND+NBD*6:ND+NBD*7, 4] = np.arange(ND+NBD*5, ND+NBD*6)[bnoidx] 
        newdart[ND+NBD*6:ND+NBD*7, 5] = np.arange(ND+NBD*4, ND+NBD*5)
        newdart[newdart[ND+NBD*5:ND+NBD*6, 6], 6] = np.arange(ND+NBD*5, ND+NBD*6)
        print(newdart)
        return DartMesh3d(newnode, newdart)

    def entity_barycenter(self, entityType, index=np.s_[:]):
        '''!
        @brief 获取每个实体的重心
        '''
        node = self.node
        if entityType=='edge':
            edge = self.ds.edge_to_node(index=index)
            bary = np.average(node[edge], axis=1)
            return bary
        elif entityType=='face':
            face, faceLoc = self.ds.face_to_node(index=index)
            NV = (faceLoc[1:] - faceLoc[:-1]).reshape(-1, 1)
            NF = len(NV)

            bary = np.zeros([NF, 3], dtype=np.float_)
            isNotOK = np.ones(NF, dtype=np.bool_)
            start = faceLoc[:-1].copy()
            while np.any(isNotOK):
                bary[isNotOK] = bary[isNotOK] + node[face[start[isNotOK]]]/NV[isNotOK]
                start[isNotOK] = start[isNotOK] + 1
                isNotOK = start < faceLoc[1:]
            return bary
        elif entityType=='cell':
            cell, cellLoc = self.ds.cell_to_node()
            NV = (cellLoc[1:] - cellLoc[:-1]).reshape(-1, 1)
            NC = self.number_of_cells()

            bary = np.zeros([NC, 3], dtype=np.float_)
            isNotOK = np.ones(NC, dtype=np.bool_)
            start = cellLoc[:-1].copy()
            while np.any(isNotOK):
                bary[isNotOK] = bary[isNotOK] + node[cell[start[isNotOK]]]/NV[isNotOK]
                start[isNotOK] = start[isNotOK] + 1
                isNotOK = start < cellLoc[1:]
            return bary[index]

    def number_of_nodes(self):
        return len(self.node)

    def number_of_edges(self):
        return len(self.ds.hedge)

    def number_of_faces(self):
        return len(self.ds.hface)

    def number_of_cells(self):
        return len(self.ds.hcell)

    def to_vtk(self, fname):
        import vtk
        import vtk.util.numpy_support as vnp

        NC = self.number_of_cells()

        node = self.node
        cell, cellLoc = self.ds.cell_to_node()
        face, faceLoc = self.ds.face_to_node()
        cell2face, cell2faceLoc = self.ds.cell_to_face()

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
        if self.nodedata:
            nodedata = self.nodedata
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

        if self.celldata:
            celldata = self.celldata
            cdata = uGrid.GetCellData()
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

    def print(self):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        print("hcell:")
        for i, val in enumerate(self.ds.hcell):
            print(i, ':', val)

        print("hedge:")
        for i, val in enumerate(self.ds.hedge):
            print(i, ":", val)

        print("dart:")
        for i, val in enumerate(self.ds.dart):
            print(i, ":", val)

        print("edge:")
        edge = self.ds.edge_to_node()
        for i, val in enumerate(edge):
            print(i, ":", val)

        print("face:")
        face, faceLoc = self.ds.face_to_node()
        for i in range(NF):
            print(i, ":", face[faceLoc[i]:faceLoc[i+1]])

        print("face2edge:")
        face2edge, face2edgeLoc = self.ds.face_to_edge()
        for i in range(NF):
            print(i, ":", face2edge[face2edgeLoc[i]:face2edgeLoc[i+1]])

        print("face2cell:")
        face2cell = self.ds.face_to_cell()
        for i, val in enumerate(face2cell):
            print(i, ":", val)

        print("cell:")
        cell, cellLoc = self.ds.cell_to_node()
        for i in range(NC):
            print(i, ":", cell[cellLoc[i]:cellLoc[i+1]])

        print("cell2edge:")
        cell2edge, cell2edgeLoc = self.ds.cell_to_edge()
        for i in range(NC):
            print(i, ":", cell2edge[cell2edgeLoc[i]:cell2edgeLoc[i+1]])

        print("cell2face:")
        cell2face, cell2faceLoc = self.ds.cell_to_face()
        for i in range(NC):
            print(i, ":", cell2face[cell2faceLoc[i]:cell2faceLoc[i+1]])

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

        f2e = f2e[:, 1]
        f2eLocation = np.zeros(NF+1, dtype=np.int_)
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

    def face_to_node(self, index=np.s_[:]):
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

        N = len(self.hface[index])
        f2nLocation = np.zeros(N+1, dtype=np.int_)
        f2nLocation[1:] = np.cumsum(NVF[index])

        f2n = np.zeros(f2nLocation[-1], dtype=np.int_) 

        current = self.hface[index].copy() #循环的dart
        idx = f2nLocation[:-1].copy() #循环的 f2n 索引
        isNotOK = np.ones(N, dtype=np.bool_)
        while np.any(isNotOK):
            f2n[idx[isNotOK]] = dart[current[isNotOK], 0]
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
        NC = len(self.hcell)

        f2c = -np.ones([NF, 4], dtype=np.int_)
        f2c[:, 0] = dart[hface, 3]
        f2c[:, 1] = dart[dart[hface, 6], 3]

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

    def edge_to_node(self, index=np.s_[:]):
        dart = self.dart
        hedge = self.hedge[index]
        NE = len(hedge)

        e2n = np.zeros([NE, 2], dtype=np.int_)
        e2n[:, 1] = dart[hedge, 0]
        e2n[:, 0] = dart[dart[hedge, 5], 0]
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
        isBDDart = dart[:, -1] == np.arange(ND)
        return isBDDart

    def boundary_node_flag(self):
        NN = len(self.hnode)
        isBDNode = np.zeros(NN, dtype=np.bool_)

        isBDDart = self.boundary_dart_flag()
        isBDNode[self.dart[isBDDart, 0]] = True
        return isBDNode

    def boundary_edge_flag(self):
        NE = len(self.hedge)
        isBDEdge = np.zeros(NE, dtype=np.bool_)

        isBDDart = self.boundary_dart_flag()
        isBDEdge[self.dart[isBDDart, 1]] = True
        return isBDEdge

    def boundary_face_flag(self):
        NF = len(self.hface)
        isBDFace = np.zeros(NF, dtype=np.bool_)

        isBDDart = self.boundary_dart_flag()
        isBDFace[self.dart[isBDDart, 2]] = True
        return isBDFace

    def boundary_cell_flag(self):
        NC = len(self.hcell)
        isBDCell = np.zeros(NC, dtype=np.bool_)

        isBDDart = self.boundary_dart_flag()
        isBDCell[self.dart[isBDDart, 3]] = True
        return isBDCell

    def boundary_dart_index(self):
        return np.where(self.boundary_dart_flag())[0]

    def boundary_node_index(self):
        return np.where(self.boundary_node_flag())[0]

    def boundary_edge_index(self):
        return np.where(self.boundary_edge_flag())[0]

    def boundary_face_index(self):
        return np.where(self.boundary_face_flag())[0]

    def boundary_cell_flag(self):
        return np.where(self.boundary_cell_flag())[0]









