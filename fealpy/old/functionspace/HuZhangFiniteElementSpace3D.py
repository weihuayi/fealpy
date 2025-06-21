import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from scipy.sparse import csr_matrix


from fealpy.functionspace.Function import Function
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.quadrature import  IntervalQuadrature
from fealpy.decorator import barycentric

class HuZhangFiniteElementSpace():
    """
    Hu-Zhang Mixed Finite Element Space 3D.
    """
    def __init__(self, mesh, p, q=None):
        self.space = LagrangeFiniteElementSpace(mesh, p, q=q) # the scalar space
        self.mesh = mesh
        self.p = p
        self.dof = self.space.dof
        self.dim = self.space.GD

        self.edof = (p-1)
        self.fdof = (p-1)*(p-2)//2
        self.cdof = (p-1)*(p-2)*(p-3)//6

        
        self.init_edge_to_dof()
        self.init_face_to_dof()
        self.init_cell_to_dof()
        self.init_orth_matrices()
        self.integralalg = self.space.integralalg
        self.integrator = self.integralalg.integrator


    def init_orth_matrices(self):
        """
        Initialize the othogonal symetric matrix basis.
        """
        mesh = self.mesh
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension()
        gdof = self.number_of_global_dofs()
        self.Tensor_Frame = np.zeros((gdof,tdim),dtype=np.float) #self.Tensor_Frame[i,:]表示第i个基函数的标架

        

        NE = mesh.number_of_edges()
        
        idx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
        TE = np.zeros((NE, 6, 6), dtype=np.float)
        self.T = np.array([
            [(1, 0, 0), (0, 0, 0), (0, 0, 0)], 
            [(0, 0, 0), (0, 1, 0), (0, 0, 0)],
            [(0, 0, 0), (0, 0, 0), (0, 0, 1)],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
            [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
            [(0, 1, 0), (1, 0, 0), (0, 0, 0)]])

        t = mesh.edge_unit_tangent() 
        _, _, frame = np.linalg.svd(t[:, np.newaxis, :]) # get the axis frame on the edge by svd
        frame[:, 0, :] = t
        for i, (j, k) in enumerate(idx):
            TE[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2
        TE[:, gdim:] *=np.sqrt(2) 

        NF = mesh.number_of_faces()
        TF = np.zeros((NF, 6, 6), dtype=np.float)
        n = mesh.face_unit_normal()
        _, _, frame = np.linalg.svd(n[:, np.newaxis, :]) # get the axis frame on the face by svd
        frame[:, 0, :] = n 
        for i, (j, k) in enumerate(idx):
            TF[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2

        TF[:, gdim:] *= np.sqrt(2) 

        



        base0 = 0

        #顶点标架
        T = np.eye(tdim,dtype=np.float)
        T[gdim:] = T[gdim:]/np.sqrt(2)
        NN = mesh.number_of_nodes()
        shape = (NN,tdim,tdim)
        self.Tensor_Frame[:NN*tdim] = np.broadcast_to(T[None,:,:],shape).reshape(-1,tdim) #顶点标架
        base0 += tdim*NN
        edof = self.edof
        #print(base0)
        if edof > 0: #边内部连续自由度标架
            base_e_continue = base0
            NE = mesh.number_of_edges()
            shape = (NE,edof,tdim-1,tdim)
            self.Tensor_Frame[base0:base0+NE*edof*(tdim-1)] = np.broadcast_to(TE[:,None,1:],shape).reshape(-1,tdim)
            base0 += NE*edof*(tdim-1)
        #print(base0)
        fdof = self.fdof
        if fdof > 0:
            NF = mesh.number_of_faces()
            shape = (NF,fdof,3,tdim)
            self.Tensor_Frame[base0:base0+NF*fdof*3] = np.broadcast_to(TF[:,None,[0,4,5]],shape).reshape(-1,tdim) #面上连续标架
            base0 +=NF*fdof*3
        #print(base0)
        #3D单元内部
        cdof = self.cdof
        if cdof > 0:
            NC = mesh.number_of_cells()
            shape = (NC,cdof,tdim,tdim)
            self.Tensor_Frame[base0:base0+NC*cdof*tdim] = np.broadcast_to(T[None,None,:,:],shape).reshape(-1,tdim) #内部标架
            base0 += NC*cdof*tdim
        #print(base0)
        if edof > 0: #边上不连续标架
            E = (gdim+1)*(gdim)//2
            NC = mesh.number_of_cells()
            cell2edge = mesh.ds.cell_to_edge()
            idx, = np.nonzero(self.dof_flags_1()[1])
            idx = self.dof_flags()[1][idx]
            idxs = np.zeros(E*edof,dtype=int)
            for i in range(E*edof):
                idxs[i], = np.nonzero(idx[i])
            self.Tensor_Frame[base0:base0+NC*E*edof] = TE[cell2edge[:,idxs],0].reshape(-1,tdim)
            base0 += NC*E*edof
        #print(base0)
        #面上不连续标架
        F = gdim+1
        cell2face = mesh.ds.cell_to_face() #(NC,F)
        idx, = np.nonzero(self.dof_flags_1()[2])
        idx = self.dof_flags()[2][idx]
        idxs = np.zeros(F*fdof,dtype=int)
        for i in range(F*fdof):
            idxs[i], = np.nonzero(idx[i])
        NC = mesh.number_of_cells()
        shape = (NC,F*fdof,3,tdim)
        self.Tensor_Frame[base0:base0+NC*F*fdof*3] = (TF[...,[1,2,3],:][cell2face[:,idxs]]).reshape(-1,tdim)
        base0 +=NC*F*fdof*3
        #print(base0)




        if edof > 0: #边内部有自由度，边界边标架需要变化
            # 对于边界边，特殊选取n0,n1,来做该边的标架
            bdEdge_index = mesh.ds.boundary_edge_index() #所有边界边指标
            Nebd = len(bdEdge_index) #边界边个数
            bdFace_index = mesh.ds.boundary_face_index() #所有边界面指标
            NFbd = len(bdFace_index) #边界面个数
            bdFace2edge = mesh.ds.face_to_edge()[bdFace_index] #面和边的关系
            bd_n = mesh.face_unit_normal()[bdFace_index] #边界面单位外法向量 #(NFbd,gdim)
            bde_t = mesh.edge_unit_tangent()[bdEdge_index] #边界边的切向
            idx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
            frame_temp = np.zeros((gdim,gdim),dtype=np.float)
            bdTE = np.zeros((tdim,tdim),dtype=np.float)

            bdedge2dof = self.edge_to_dof()[bdEdge_index]
            #print(bdedge2dof.shape)

            self.Cross_edge_index = [] # 存储交边与其对应的边界面指标，用list来存储，规模不大
                                        #角点标架先不改变，根据边界条件来选取角点标架
            self.Cross_edge_bdFace_index = [] #对应的边界面指标

            shape = (edof,tdim-1,tdim)
            for i in range(Nebd):
                index, = np.nonzero(np.sum(bdFace2edge==bdEdge_index[i],axis=-1))
                indexs = np.array([index[0]],dtype=int)
                index = np.delete(index,0)
                while len(index)>0:
                    if np.min(np.max(np.abs(bd_n[index[0]][None,:]-bd_n[indexs]),axis=-1))>1e-15:
                        #说明当前法向与现有的法向不重合
                        indexs = np.append(indexs,index[0])
                    index = np.delete(index,0)

                if len(indexs) == 1: 
                    #该边对应的面的法向都是一致的,将该法向选为一个方向
                    bde_t_temp = bde_t[i]
                    bdn_temp = bd_n[indexs]
                    bdn1_temp  = np.cross(bde_t_temp,bdn_temp)
                    frame_temp[0] = bde_t_temp
                    frame_temp[1] = bdn_temp #此分量为边界外法向
                    frame_temp[2] = bdn1_temp
                    for i_temp, (j_temp, k_temp) in enumerate(idx):
                        bdTE[i_temp] = (frame_temp[j_temp, idx[:, 0]]*frame_temp[k_temp, idx[:, 1]] 
                                        + frame_temp[j_temp, idx[:, 1]]*frame_temp[k_temp, idx[:, 0]])/2
                    bdTE[gdim:] *=np.sqrt(2)
                    #连续标架
                    base0 = base_e_continue + bdEdge_index[i]*edof*(tdim-1)
                    #print(np.min(bdedge2dof[i][1:4,1:]),base0)
                    self.Tensor_Frame[base0:base0+edof*(tdim-1)] = np.broadcast_to(bdTE[None,1:],shape).reshape(-1,tdim)
                    #不连续标架不用变,其对应仍未t*t.T
                elif len(indexs) == 2:
                    self.Cross_edge_index.append(bdEdge_index[i])
                    self.Cross_edge_bdFace_index.append(bdFace_index[indexs])
                else:
                    raise ValueError("Warn: The geometry shape is complex, and there are more than two boundary face related to the cross edge！")








        # 对于边界顶点，如果只有一个外法向，选取该外法向来做标架
        bdNode_index = mesh.ds.boundary_node_index() #所有边界顶点点指标
        Nnbd = len(bdNode_index) #边界顶点个数
        bdFace_index = mesh.ds.boundary_face_index() #所有边界面指标
        NFbd = len(bdFace_index) #边界面个数
        bdFace2node = mesh.entity('face')[bdFace_index] #边界面到点的对应 (NFbd,gdim)
        bd_n = mesh.face_unit_normal()[bdFace_index] #边界单位外法向量 #(NFbd,gdim)

        _, _, frame = np.linalg.svd(bd_n[:, np.newaxis, :]) # get the axis frame on the face by svd
        #3D case 保证取法与面标架取法一致
        idx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
        frame[:, 0, :] = bd_n 

        bdTF = np.zeros((NFbd, tdim, tdim), dtype=np.float)
        for i, (j, k) in enumerate(idx):
            bdTF[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2
        bdTF[:, gdim:] *=np.sqrt(2)

        #print(frame[84,0,0],bdFace_index[84])


        
        self.Corner_point_index = [] #存储角点与其对应的边界面指标，用list来存储，规模不大
                                     #角点标架先不改变，根据边界条件来选取角点标架
        self.Corner_point_bdFace_index = [] #对应的边界面指标
        for i in range(Nnbd):
            index, = np.nonzero(np.sum(bdFace2node==bdNode_index[i],axis=-1)) #找到对应到边界面
            indexs = np.array([index[0]],dtype=int)
            index = np.delete(index,0)

            while len(index)>0:
                if np.min(np.max(np.abs(bd_n[index[0]][None,:]-bd_n[indexs]),axis=-1))>1e-15:
                    #说明当前法向与现有的法向不重合
                    indexs = np.append(indexs,index[0])
                index = np.delete(index,0)
                
            if len(indexs) == 1:
                self.Tensor_Frame[tdim*bdNode_index[i]:tdim*bdNode_index[i]+tdim]=bdTF[indexs[0]]
            else:
                self.Corner_point_index.append(bdNode_index[i])
                self.Corner_point_bdFace_index.append(bdFace_index[indexs])


    def __str__(self):
        return "Hu-Zhang mixed finite element space 3D!"

    def number_of_global_dofs(self):
        """
        """
        p = self.p
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension()

        mesh = self.mesh

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = tdim*NN

        if p > 1:
            edof = p - 1
            NE = mesh.number_of_edges()
            gdof += (tdim-1)*edof*NE # 边内部连续自由度的个数 
            E = mesh.number_of_edges_of_cells() # 单元边的个数
            gdof += NC*E*edof # 边内部不连续自由度的个数 

        if p > 2:
            fdof = (p+1)*(p+2)//2 - 3*p # 面内部自由度的个数
            NF = mesh.number_of_faces()
            gdof += 3*fdof*NF # 面内部连续自由度的个数
            F = mesh.number_of_faces_of_cells() # 每个单元面的个数
            gdof += 3*F*fdof*NC # 面内部不连续自由度的个数

        if p > 3:
            ldof = self.dof.number_of_local_dofs()
            V = mesh.number_of_nodes_of_cells() # 单元顶点的个数
            cdof = ldof - E*edof - F*fdof - V 
            gdof += tdim*cdof*NC

        return gdof 

    def number_of_local_dofs(self):
        tdim = self.tensor_dimension() 
        ldof = self.dof.number_of_local_dofs()
        return tdim*ldof

    def cell_to_dof(self):
        return self.cell2dof

    def face_to_dof(self):
        return self.face2dof

    def edge_to_dof(self):
        return self.edge2dof

    def init_cell_to_dof(self):
        """
        构建局部自由度到全局自由度的映射矩阵

        Returns
        -------
        cell2dof : ndarray with shape (NC, ldof*tdim)
            NC: 单元个数
            ldof: p 次标量空间局部自由度的个数
            tdim: 对称张量的维数
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() # 张量维数
        p = self.p
        dof = self.dof # 标量空间自由度管理对象 
       
        c2d = dof.cell2dof[..., np.newaxis]
        ldof = dof.number_of_local_dofs() # ldof : 标量空间单元上自由度个数
        cell2dof = np.zeros((NC, ldof, tdim), dtype=np.int) # 每个标量自由度变成 tdim 个自由度
        base0 = 0
        base1 = 0

        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来
        idx, = np.nonzero(dofFlags[0]) # 局部顶点自由度的编号
        cell2dof[:, idx, :] = tdim*c2d[:, idx] + np.arange(tdim)
        base1 += tdim*NN # 这是张量自由度编号的新起点
        base0 += NN # 这是标量编号的新起点

        #print(np.max(cell2dof),base1)

        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        if len(idx) > 0:
            #  0号局部自由度对应的是切向不连续的自由度, 留到后面重新编号
            cell2dof[:, idx, 1:] = base1 + (tdim-1)*(c2d[:, idx] - base0) + np.arange(tdim - 1)
            edof = self.edof
            base1 += (tdim-1)*edof*NE
            base0 += edof*NE

        #print(np.max(cell2dof),base1)

        idx, = np.nonzero(dofFlags[2])
        if len(idx) > 0:            
            # 1, 2, 3 号局部自由度对应切向不连续的张量自由度, 留到后面重新编号
            # TODO: check it is right
            cell2dof[:, idx.reshape(-1, 1), np.array([0, 4, 5])]= base1 + (tdim - 3)*(c2d[:, idx] - base0) + np.arange(tdim - 3)
            fdof = self.fdof
            NF = mesh.number_of_faces()
            base1 += (tdim - 3)*fdof*NF
            base0 += fdof*NF 

        #print(np.max(cell2dof),base1) 

        idx, = np.nonzero(dofFlags[3])          
        if len(idx) > 0:
            cell2dof[:, idx, :] = base1 + tdim*(c2d[:, idx] - base0) + np.arange(tdim)
            cdof = self.cdof
            base1 += tdim*cdof*NC
        #print(np.max(cell2dof),base1) 
        idx, = np.nonzero(dofFlags[1])
        if len(idx) > 0:
            cell2dof[:, idx, 0] = base1 + np.arange(NC*len(idx)).reshape(NC, len(idx))
            base1+=NC*len(idx)
        #print(np.max(cell2dof),base1) 


        idx, = np.nonzero(dofFlags[2])
        #print(idx)
        if len(idx) > 0:
            cell2dof[:, idx.reshape(-1, 1), np.array([1, 2, 3])] = base1 + np.arange(NC*len(idx)*3).reshape(NC, len(idx), 3)
            base1+=NC*len(idx)*3
        #self.cell2dof = cell2dof.reshape(NC, -1)
        self.cell2dof = cell2dof
        #print(np.max(self.cell2dof),base1)

        
    def init_face_to_dof(self):
        """
        构建局部自由度到全局自由度的映射矩阵

        Returns
        -------
        face2dof : ndarray with shape (NF, ldof*tdim)
            NF: 单元个数
            ldof: p 次标量空间局部自由度的个数
            tdim: 对称张量的维数
        """
        gdim = self.geo_dimension()
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()                    
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() # 张量维数
        p = self.p
        dof = self.dof # 标量空间自由度管理对象

        f2d = dof.face_to_dof()[...,np.newaxis]
        ldof = dof.number_of_local_dofs(doftype='face')

        face2dof = np.zeros((NF,ldof,tdim),dtype=np.int)-1

        dofFlags = self.face_dof_falgs_1() # 把不同类型的自由度区分开来
        idx, = np.nonzero(dofFlags[0]) # 局部顶点自由度的编号
        face2dof[:,idx,:] = tdim*f2d[:, idx] + np.arange(tdim)

        base0 = 0
        base1 = 0
        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        if len(idx) > 0:
            base0 += NN # 这是标量编号的新起点
            base1 += tdim*NN # 这是张量自由度编号的新起点
            #  0号局部自由度对应的是切向不连续的自由度, 此时点乘外法向量为0,不算是边界,已经被标记为-1
            face2dof[:, idx, 1:] = base1 + (tdim-1)*(f2d[:, idx] - base0) + np.arange(tdim - 1)

        idx, = np.nonzero(dofFlags[2]) #面内部自由度编号
        if len(idx) > 0:
            edof = p - 1
            base0 += edof*NE
            base1 += (tdim-1)*edof*NE
            # 1, 2, 3 号局部自由度对应切向不连续的张量自由度, 此时点乘外法向量为0,不算是边界
            face2dof[:, idx.reshape(-1, 1), np.array([0, 4, 5])]= base1 + (tdim - 3)*(f2d[:, idx] - base0) + np.arange(tdim - 3)

        self.face2dof = face2dof

    def init_edge_to_dof(self):
            mesh = self.mesh
            NN = mesh.number_of_nodes()
            NE = mesh.number_of_edges()
            NC = mesh.number_of_cells()                    
            gdim = self.geo_dimension()
            tdim = self.tensor_dimension() # 张量维数
            p = self.p
            dof = self.dof # 标量空间自由度管理对象
            e2d = dof.edge_to_dof()[...,np.newaxis]
            ldof = dof.number_of_local_dofs(doftype='edge')
            edge2dof = np.zeros((NE,ldof,tdim),dtype=np.int)-1# 每个标量自由度变成 tdim 个自由度

            dofFlags = self.edge_dof_falgs_1() # 把不同类型的自由度区分开来
            idx, = np.nonzero(dofFlags[0])# 局部顶点自由度的编号
            edge2dof[:,idx,:] = tdim*e2d[:,idx] + np.arange(tdim)

            base0 = 0
            base1 = 0
            idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
            if len(idx)>0:
                base0 += NN # 这是标量编号的新起点
                base1 += tdim*NN # 这是张量自由度编号的新起点
                #  0号局部自由度对应的是切向不连续的自由度, 此部分点乘外法向量为0,不算是边界，可以不用编号，已经被标记为-1
                edge2dof[:,idx,1:] = base1+(tdim-1)*(e2d[:, idx] - base0) + np.arange(tdim - 1)

            self.edge2dof = edge2dof


    def geo_dimension(self):
        return self.dim

    def tensor_dimension(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def dof_flags(self):
        """ 对标量空间中的自由度进行分类, 分为边内部自由度, 面内部自由度(如果是三维空间的话)及其它自由度 

        Returns
        -------

        isOtherDof : ndarray, (ldof,)
            除了边内部和面内部自由度的其它自由度
        isEdgeDof : ndarray, (ldof, 3) or (ldof, 6) 
            每个边内部的自由度
        isFaceDof : ndarray, (ldof, 4)
            每个面内部的自由度
        -------

        """
        dim = self.geo_dimension()
        dof = self.dof 
        
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0 # 
        isOtherDof = (~isEdgeDof0) # 除了边内部自由度之外的其它自由度
                                   # dim = 2: 包括点和面内部自由度
                                   # dim = 3: 包括点, 面内部和体内部自由度
        if dim == 2:
            return isOtherDof, isEdgeDof
        elif dim == 3:
            isFaceDof = dof.is_on_face_local_dof()
            isFaceDof[isPointDof, :] = False
            isFaceDof[isEdgeDof0, :] = False

            isFaceDof0 = np.sum(isFaceDof, axis=-1) > 0
            isOtherDof = isOtherDof & (~isFaceDof0) # 三维情形下, 从其它自由度中除去面内部自由度

            return isOtherDof, isEdgeDof, isFaceDof
        else:
            raise ValueError('`dim` should be 2 or 3!')

    def dof_flags_1(self):
        """ 
        对标量空间中的自由度进行分类, 分为:
            点上的自由由度
            边内部的自由度
            面内部的自由度
            体内部的自由度

        Returns
        -------

        """
        gdim = self.geo_dimension() # the geometry space dimension
        dof = self.dof 
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0

        isFaceDof = dof.is_on_face_local_dof()
        isFaceDof[isPointDof, :] = False
        isFaceDof[isEdgeDof0, :] = False

        isFaceDof0 = np.sum(isFaceDof, axis=-1) > 0
        return isPointDof, isEdgeDof0, isFaceDof0, ~(isPointDof | isEdgeDof0 | isFaceDof0)

    def face_dof_falgs(self):
        """
        对标量空间中面上的基函数自由度进行分类，分为：
            点上的自由由度
            边内部的自由度
            面内部的自由度        
        """
        p = self.p
        gdim = self.geo_dimension()
        TD = 2
        multiIndex = self.space.multi_index_matrix[TD](p)#(ldof,3)
        isPointDof = (np.sum(multiIndex == p, axis=-1) > 0)
        isEdgeDof = (multiIndex == 0)
        isEdgeDof[isPointDof] = False
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) >0
        return isPointDof, isEdgeDof, ~(isPointDof | isEdgeDof0)

    def face_dof_falgs_1(self):
        """
        对标量空间中面上的基函数自由度进行分类，分为：
            点上的自由由度
            边内部的自由度
            面内部的自由度        
        """
        p = self.p
        gdim = self.geo_dimension()
        TD = 2
        multiIndex = self.space.multi_index_matrix[TD](p)#(ldof,3)
        isPointDof = (np.sum(multiIndex == p, axis=-1) > 0)
        isEdgeDof = (multiIndex == 0)
        isEdgeDof[isPointDof] = False

        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) >0
        return isPointDof, isEdgeDof0, ~(isPointDof | isEdgeDof0)

    def edge_dof_falgs(self):
        """
        对标量空间中边上的基函数自由度进行分类，分为：
            点上的自由由度 
        """ 
        p = self.p
        TD = 1
        multiIndex = self.space.multi_index_matrix[TD](p)#(ldof,2)
        isPointDof = (np.sum(multiIndex == p, axis=-1) > 0)    
        isEdgeDof0  = ~isPointDof
        return isPointDof, isEdgeDof0    

    def edge_dof_falgs_1(self):
        """
        对标量空间中边上的基函数自由度进行分类，分为：
            点上的自由由度 
        """ 
        p = self.p
        TD = 1
        multiIndex = self.space.multi_index_matrix[TD](p)#(ldof,2)
        isPointDof = (np.sum(multiIndex == p, axis=-1) > 0)    
        isEdgeDof0  = ~isPointDof
        return isPointDof, isEdgeDof0    

    @barycentric
    def face_basis(self,bc):
        gdim = self.geo_dimension()
        phi0 = self.space.face_basis(bc) #(NQ,1,ldof)   
        face2dof = self.face2dof#(NF,ldof,tdim)
        phi = np.einsum('nijk,...ni->...nijk',self.Tensor_Frame[face2dof],phi0) #(NF,ldof,tdim,tdim), (NQ,ldof)
        #在不连续标架算出的结果不对，但是不影响，因为其自由度就是定义在单元体上的
        #不连续标架有:边内部第0个标架，面内部第1,2,3个标架
        return phi #(NQ,NF,ldof,tdim,tdim) 


    @barycentric
    def edge_basis(self,bc):
        phi0 = self.space.face_basis(bc) #(NQ,1,ldof)       
        edge2dof = self.edge2dof #(NC,ldof,tdim)
        phi = np.einsum('nijk,...ni->...nijk',self.Tensor_Frame[edge2dof],phi0) #(NE,ldof,tdim,tdim), (NQ,1,ldof)
        #在不连续标架算出的结果不对，但是不影响，因为其自由度就是定义在单元体上的
        #不连续标架有:边内部第0个标架
        return phi #(NQ,NE,ldof,tdim,tdim)  


    @barycentric
    def basis(self, bc, index=np.s_[:]):
        """
        Parameters
        ----------
        bc : ndarray with shape (NQ, dim+1)
            bc[i, :] is i-th quad point
        index : ndarray
            有时我我们只需要计算部分单元上的基函数
        Returns
        -------
        phi : ndarray with shape (NQ, NC, ldof, tdim, 3 or 6)
            NQ: 积分点个数
            NC: 单元个数
            ldof: 标量空间的单元自由度个数
            tdim: 对称张量的维数
        """
        mesh = self.mesh

        gdim = self.geo_dimension() 
        tdim = self.tensor_dimension()
        phi0 = self.space.basis(bc) #(NQ,1,ldof)
        cell2dof = self.cell2dof[index] #(NC,ldof,tdim)

        phi = np.einsum('nijk,...ni->...nijk',self.Tensor_Frame[cell2dof],phi0) #(NC,ldof,tdim,tdim), (NQ,1,ldof)
        #print(phi.shape)
        return phi  #(NQ,NC,ldof,tdim,tdim) 最后一个维度表示tensor


    @barycentric
    def div_basis(self, bc, index=np.s_[:]):
        mesh = self.mesh

        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() 

        # the shape of `gphi` is (NQ, NC, ldof, gdim)
        gphi = self.space.grad_basis(bc, index=index) 
        cell2dof = self.cell2dof[index]
        shape = list(gphi.shape)
        shape.insert(-1, tdim)
        # the shape of `dphi` is (NQ, NC, ldof, tdim, gdim)

        VAL = np.einsum('iljk,kmn->iljmn',self.Tensor_Frame[cell2dof],self.T) #(NC,ldof,tdim,gdim,gdim)
        dphi = np.einsum('...ikm,ikjmn->...ikjn',gphi,VAL) #(NQ,NC,ldof,gdim), (NC,ldof,tdim,gdim,gdim)
        #print(dphi.shape)


        return dphi #(NQ,NC,ldof,tdim,gdim)


    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        uh = uh[cell2dof[index]]
        val = np.einsum('...ijkm, ijk->...im', phi, uh) #(NQ,NC,tdim)
        val = np.einsum('...k, kmn->...mn', val, self.T)
        return val #(NQ,NC,gdim,gdim)


    def compliance_tensor_matrix(self,mu=1,lam=1):
        ldof = self.number_of_local_dofs()
        tdim = self.tensor_dimension()
        gdim = self.geo_dimension()
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        NC = self.mesh.number_of_cells()
        NQ = bcs.shape[0]

        phi = self.basis(bcs).reshape(NQ,NC,-1,tdim)#(NQ,NC,ldof,tdim)
        #compliance_tensor
        aphi = phi.copy()
        t = np.sum(aphi[..., 0:gdim], axis=-1)
        aphi[..., 0:gdim] -= lam/(2*mu+gdim*lam)*t[..., np.newaxis]
        aphi /= 2*mu

        #construct matrix
        d = np.array([1, 1, 1, 2, 2, 2])
        M = np.einsum('i, ijkm, m, ijom, j->jko', ws, aphi, d, phi, self.mesh.entity_measure(), optimize=True)

        I = np.einsum('ij, k->ijk', self.cell2dof.reshape(NC,-1), np.ones(ldof))
        J = I.swapaxes(-1, -2)
        tgdof = self.number_of_global_dofs()

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(tgdof, tgdof))

        return M



    def  parallel_compliance_tensor_matrix(self,mu=1,lam=1):
        ldof = self.number_of_local_dofs()
        tgdof = self.number_of_global_dofs()
        tdim = self.tensor_dimension()
        gdim = self.geo_dimension()
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        NC = self.mesh.number_of_cells()
        Tmeasure = self.mesh.entity_measure()

        basis0 = self.basis
        cell2dof0 = self.cell_to_dof().reshape(NC,-1)
        gdof0 = tgdof

        gdof1 = gdof0

        # 对问题进行分割
        nc = mp.cpu_count()-2

        block = NC//nc
        r = NC%nc

        #print(NC,nc,block,r)

        index = np.full(nc+1,block)
        index[0] = 0
        index[1:r+1] +=1
        np.cumsum(index,out=index)

        M = csr_matrix((gdof0,gdof1))
        M1 = csr_matrix((gdof0,gdof1))

        def f(i):
            s = slice(index[i],index[i+1])
            measure = Tmeasure[s]
            c2d0 = cell2dof0[s]
            c2d1 = c2d0
            shape = (len(measure),c2d0.shape[1],c2d1.shape[1]) #（lNC,ldof0,ldof1)
            lNC  = index[i+1]-index[i]

            d = np.array([1, 1, 1, 2, 2, 2])
            Mi = np.zeros(shape,measure.dtype)
            for bc, w in zip(bcs,ws):
                phi0 = basis0(bc,index=s).reshape(lNC,-1,tdim) #(lNC,ldof,tdim)
                t = np.sum(phi0[...,:gdim],axis=-1)
                Mi +=np.einsum('jkl,l,jol,j->jko',phi0, d, phi0, w*measure)
                Mi -=lam*np.einsum('jk,jo,j->jko',t,t,w*measure)/(2*mu+lam*gdim)

            Mi /= 2*mu
            I = np.broadcast_to(c2d0[:, :, None], shape=Mi.shape)
            J = np.broadcast_to(c2d1[:, None, :], shape=Mi.shape)

            Mi = csr_matrix((Mi.flat, (I.flat, J.flat)), shape=(gdof0,gdof1))

            return Mi


    
        
        # 并行组装总矩阵
        with Pool(nc) as p:
            Mi= p.map(f, range(nc))

        for val in Mi:
            M += val

        return M






    def div_matrix(self,vspace):
        '''

        Notes
        -----
        (div tau, v)

        gdim == 3
        v= [[phi,0,0],[0,phi,0],[0,0,phi]]

        [[B0],[B1],[B2]]

        '''
        tldof = self.number_of_local_dofs()
        vldof = vspace.number_of_local_dofs()
        tgdof = self.number_of_global_dofs()
        vgdof = vspace.number_of_global_dofs()
        
        gdim = self.geo_dimension()
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        NC = self.mesh.number_of_cells()
        NQ = bcs.shape[0]

        dphi = self.div_basis(bcs).reshape(NQ,NC,-1,gdim) #(NQ, NC, ldof*tdim, gdim)
        vphi = vspace.basis(bcs)#(NQ,1,vldof)


        B0 = np.einsum('i,ijk,ijo,j->jko',ws,vphi,dphi[...,0],self.mesh.entity_measure(), optimize=True)
        B1 = np.einsum('i,ijk,ijo,j->jko',ws,vphi,dphi[...,1],self.mesh.entity_measure(), optimize=True)
        B2 = np.einsum('i,ijk,ijo,j->jko',ws,vphi,dphi[...,2],self.mesh.entity_measure(), optimize=True)



        I = np.einsum('ij, k->ijk', vspace.cell_to_dof(), np.ones(tldof,dtype=int))
        J = np.einsum('ij, k->ikj', self.cell_to_dof().reshape(NC,-1,), np.ones(vldof,dtype=int))   

        B0 = csr_matrix((B0.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))
        B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))
        B2 = csr_matrix((B2.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))

        return B0, B1, B2


    def parallel_div_matrix(self,vspace):
        '''
        把网格中的单元分组，再分组组装相应的矩阵。对于三维大规模问题，如果同时计
        算所有单元的矩阵，占用内存会过多，效率过低。
        '''

        tldof = self.number_of_local_dofs()
        vldof = vspace.number_of_local_dofs()
        tgdof = self.number_of_global_dofs()
        vgdof = vspace.number_of_global_dofs()
        gdim = self.geo_dimension()
        bcs, ws = self.integrator.quadpts, self.integrator.weights
        NC = self.mesh.number_of_cells()
        Tmeasure = self.mesh.entity_measure()
        




        basis1 = self.div_basis
        cell2dof1 = self.cell_to_dof().reshape(NC,-1)
        gdof1 = tgdof

        basis0 = vspace.basis
        cell2dof0 = vspace.cell_to_dof()
        gdof0 = vgdof




        # 对问题进行分割
        nc = mp.cpu_count()-2

        block = NC//nc
        r = NC%nc

        #print(NC,nc,block,r)

        index = np.full(nc+1,block)
        index[0] = 0
        index[1:r+1] +=1
        np.cumsum(index,out=index)
        #print(index)

        B0 = csr_matrix((gdof0, gdof1))
        B1 = csr_matrix((gdof0, gdof1))
        B2 = csr_matrix((gdof0, gdof1))

        def f(i):
            s = slice(index[i],index[i+1])
            measure = Tmeasure[s]
            c2d0 = cell2dof0[s]
            c2d1 = cell2dof1[s]

            shape = (len(measure),c2d0.shape[1],c2d1.shape[1]) #（lNC,ldof0,ldof1)
            lNC  = index[i+1]-index[i]
            M0 = np.zeros(shape,measure.dtype)
            M1 = np.zeros(shape,measure.dtype)
            M2 = np.zeros(shape,measure.dtype)
            for bc, w in zip(bcs, ws):
                phi0 = basis0(bc,index=s)#(1,vldof)
                phi1 = basis1(bc,index=s).reshape(lNC,-1,gdim)#(lNC, ldof*tdim, gdim)
                M0 += np.einsum('jk,jo,j->jko',phi0, phi1[...,0], w*measure)
                M1 += np.einsum('jk,jo,j->jko',phi0, phi1[...,1], w*measure)
                M2 += np.einsum('jk,jo,j->jko',phi0, phi1[...,2], w*measure)


            I = np.broadcast_to(c2d0[:, :, None], shape=M0.shape)
            J = np.broadcast_to(c2d1[:, None, :], shape=M0.shape)

            Bi0 = csr_matrix((M0.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            Bi1 = csr_matrix((M1.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            Bi2 = csr_matrix((M2.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            return Bi0,Bi1,Bi2

        # 并行组装总矩阵

        with Pool(nc) as p:
            Bi= p.map(f, range(nc))
            
        for val in Bi:
            B0 += val[0]
            B1 += val[1]
            B2 += val[2]
        

        return B0, B1, B2





    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dphi = self.div_basis(bc, index=index) #(NQ,NC,ldof,tdim,gdim)
        cell2dof = self.cell_to_dof()
        uh = uh[cell2dof[index]]
        val = np.einsum('...ijkm, ijk->...im', dphi, uh)
        return val #(NQ,NC,gdim)

    def interpolation(self, u):
        ipoint = self.dof.interpolation_points()
        c2d = self.dof.cell2dof
        val = u(ipoint)[c2d][:,:,None] #u 是一个tensor val.shape = (NC,ldof,1,gdim,gidm)
        cell2dof = self.cell2dof #(NC,ldof,tdim)

        uI = Function(self)
        Tensor_Frame = np.einsum('...k,kmn->...mn',self.Tensor_Frame,self.T) #(gdof,gdim,gdim)
        uI[cell2dof] = np.einsum('...mn,...mn->...',Tensor_Frame[cell2dof],val)
        return uI
 
        def function(self, dim=None):
            f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)

    def function(self, dim=None):
        f = Function(self)
        return f


    def set_essential_bc(self, uh, gN, M, B0, B1, B2, F0, threshold=None):
        """
        初始化压力的本质边界条件，插值一个边界sigam,使得sigam*n=gN,对于角点，相交边，要小心选取标架
        由face2bddof 形状为(NFbd,ldof,tdim)
        3Dcase时 面标架:face2bddof[...,0]--法向标架， face2bddof[...,1]--切向0标架， face2bddof[...,2]--切向1标架
                       face2bddof[...,3]--切0切1标架组合， face2bddof[...,4]--法向,切向0标架， face2bddof[...,5]--法向,切向1标架
        """
        mesh = self.mesh
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension()
        ipoint = self.dof.interpolation_points()
        gdof = self.number_of_global_dofs()
        node = mesh.entity('node')
        edge = mesh.entity('edge')


        if type(threshold) is np.ndarray:
            index = threshold #此种情况后面补充
        else:
            if threshold is  not None:
                index = mesh.ds.boundary_face_index()
                #print(index.shape)
                bc = mesh.entity_barycenter('face',index=index)
                flag = threshold(bc) #(3,gNbd) 第0行表示给的法向投影，第1,2行分量表示给的切向投影
                flag_idx = (np.sum(flag,axis=0)>0) #(gNFbd,)
                index = index[flag_idx] #(NFbd,)
                NFbd = len(index)

                bd_index_type = np.zeros((3,NFbd),dtype=np.bool_)
                bd_index_type[0] = flag[0][flag_idx] #第0个分量表示给的法向投影
                bd_index_type[1] = flag[1][flag_idx] #第1个分量表示给的切向投影
                bd_index_type[2] = flag[2][flag_idx] #第2个分量表示给的切向投影
                
                n = mesh.face_unit_normal()[index] #(NEbd,gdim)
                _, _, frame = np.linalg.svd(n[:, np.newaxis, :]) # get the axis frame on the face by svd
                frame[:,0] = n
                t0 = frame[:,1]
                t1 = frame[:,2]
                isBdDof = np.zeros(gdof,dtype=np.bool_)#判断是否为固定顶点
                Is_inner_face_idx = np.ones(NFbd,dtype=np.bool_)#找出内边
                f2bd = self.dof.face_to_dof()[index]#(NFbd,ldof)
                ipoint = ipoint[f2bd] #(NFbd,ldof,gdim)
                facebd2dof = self.face2dof[index] #(NFbd,ldof,tdim)
                

                val = gN(ipoint,n[...,None,:],t0=t0[...,None,:],t1=t1[...,None,:]) #(NFbd,ldof,gdim)

                #print(val[0])
                #print(ipoint[0])
                #self.mypoint = ipoint
                #print(ipoint)
                #print(val.shape)

                #TODO 边界插值处理，将点, 边内部，面内部自由度分别处理。
                self.set_essential_bc_inner_face(facebd2dof,bd_index_type,frame,val,uh,isBdDof) #处理所有边界面内部点
                self.set_essential_bc_inner_edge(index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,B2,F0) #处理所有边界边内部点
                self.set_essential_bc_vertex(index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,B2,F0) #处理所有边界顶点


        return isBdDof









    def set_essential_bc_inner_face(self,facebd2dof,bd_index_type,frame,val,uh,isBdDof):
        inner_face_dof, = np.nonzero(self.face_dof_falgs_1()[2])
        #print(inner_face_dof)
        bdface = facebd2dof[:,inner_face_dof][:,:,[0,5,4]] #(NFbd,fdof,3)
        bdTensor_Frame = self.Tensor_Frame[bdface] #(NFbd,fdof,3,tdim)
        val_temp = val[:,inner_face_dof] #(NFbd,3,gdim) frame (NFbd,3,gdim)
        n = frame[:,0]
        for i in range(3):
            bd_index_temp, = np.nonzero(bd_index_type[i])
            bdTensor_Frame_projection = np.einsum('ijl,lmn,in,im->ij',bdTensor_Frame[:,:,i,:],self.T,frame[:,i],n)
            #(NFbd,fdof)
            val_projection = np.einsum('ijk,ik->ij',val_temp,frame[:,i]) #(NFbd,fdof)
            uh[bdface[bd_index_temp,:,i]] = val_projection[bd_index_temp]/bdTensor_Frame_projection[bd_index_temp]
            isBdDof[bdface[bd_index_temp,:,i]] = True


        

    def set_essential_bc_inner_edge(self,index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,B2,F0):
        edof = self.edof
        if edof > 0:
            tdim = self.tensor_dimension()
            gdim = self.geo_dimension()
            edge_dof_flag = self.face_dof_falgs()[1] #(ldof,3)
            edge_dof_flag_index = np.zeros((3,edof),dtype=int) #(3,edof)
            for i in range(3):
                edge_dof_flag_index[i], = np.nonzero(edge_dof_flag[:,i])
            #print(edge_dof_flag_index)      
            NFbd = len(index)
            bd_faceedge2dof = np.zeros((NFbd,3,edof,tdim-1),dtype=np.int)
            bd_val = np.zeros((NFbd,3,edof,gdim),dtype=np.float)

            for i in range(3):
                inner_edge_dof = edge_dof_flag_index[i]
                bd_faceedge2dof[:,i] = facebd2dof[:,inner_edge_dof][...,1:]#(NFbd,3,edof,tdim-1)
                #print(bd_faceedge2dof[:,i]) 
                bd_val[:,i] = val[:,inner_edge_dof] #(NFbd,3,edof,gdim)
                #print('\n\n',bd_val[0,i])
            #print(bd_val[0,0])


            NN = self.mesh.number_of_nodes()
            base1 = tdim*NN
            bd_face2edge = np.array((bd_faceedge2dof[:,:,0,0]-base1)/((tdim-1)*edof),dtype=int)#(NFbd,3)
            #print(bd_face2edge[0])
            #print(bd_faceedge2dof[0,0])
            bdedge = np.unique(bd_face2edge)#boundary edge index

            #print(bdedge)


            Cross_edge_index_all = np.array(self.Cross_edge_index,dtype=int) #所有相交边
            bdedge = np.setdiff1d(bdedge,Cross_edge_index_all)
            INEbd = len(bdedge) #边界非相交边个数
            edge2face_idx = np.zeros((INEbd,2),dtype=int) #(INEbd,2)
            #print(bdedge)


            #######################################
            #边界内部边点插值
            bd_face2edge = bd_face2edge.T.reshape(-1) #(NFbd*3,)
            idx = (bdedge[:,None]==bd_face2edge)
            bd_face2edge = bd_face2edge.reshape((3,NFbd)).T #(NFbd,3)
            for i in range(INEbd):
                edge2face_idx[i], = np.nonzero(idx[i])
            edge2face_idx_idx = edge2face_idx//NFbd #表示该边在对应面中的顺序
            edge2face_idx = np.mod(edge2face_idx,NFbd)

            #print(edge2face_idx[0])
            #print(bd_face2edge[edge2face_idx[0]])
            #print(edge2face_idx_idx[0])
            #print(bd_val[edge2face_idx[0,0],edge2face_idx_idx[0,0]])
            
            bd_dof = bd_faceedge2dof[edge2face_idx[:,0],edge2face_idx_idx[:,0]][...,[0,2,4]]#(INEbd,edof,3)
            bdedge_index_type = bd_index_type[:,edge2face_idx[:,0]]#(3,INEbd) 边界自由度类型
            val_temp = bd_val[edge2face_idx[:,0],edge2face_idx_idx[:,0]] #(INEbd,edof,gdim)

            #print(val_temp[0])

            frame_temp = frame[edge2face_idx[:,0]]#(INEbd,3,gdim)
            bd_t = self.mesh.edge_unit_tangent()[bdedge]
            frame_temp[:,2] = bd_t
            frame_temp[:,1] = np.cross(bd_t,frame_temp[:,0])
            val_temp = np.einsum('ijk,ilk->ijl',val_temp,frame_temp)#(INEbd,edof,3)
            bdTensor_Frame = self.Tensor_Frame[bd_dof] #(INEbd,edof,3,tdim)
            T = self.T
            bdTn = np.einsum('ijkl,lmn,im,ikn->ijk',bdTensor_Frame,T,frame_temp[:,0],frame_temp) #(INEbd,edof,3)
            val_temp = val_temp/bdTn
            for i in range(3): #非三个边界条件都给定时可能出错
                idx, = np.nonzero(bdedge_index_type[i])
                if len(idx)>0:
                    uh[bd_dof[idx,:,i]] = val_temp[idx,:,i]
                    isBdDof[bd_dof[idx,:,i]] = True

            #####################################
            #边界相交边插值，数量较少，逐一处理即可
            n = frame[:,0] #(NFbd,)
            t = self.mesh.edge_unit_tangent()
            Cross_edge_to_face_index_all = np.array(self.Cross_edge_bdFace_index,dtype=int)
            Correct_edge_index = []
            Cross_edge_index = []

            for i in range(len(Cross_edge_index_all)):
                cross_edge = Cross_edge_index_all[i]
                cross_edge_to_face_index = Cross_edge_to_face_index_all[i]#一个1维数组，3D情况下只有两个分量
                idx, = np.nonzero(np.sum(index[:,None] == cross_edge_to_face_index,axis=-1))#查看该边界边是否是nuemann边界
                if len(idx) == 1: #此时只有一边是Neumann边界，需要变换标架，按照该边界面来投影
                    Correct_edge_index.append([cross_edge,idx])
                elif len(idx) == 2:
                    Cross_edge_index.append([cross_edge,idx])


            ##此时只有一个面是Neumann边界，需要变换标架，按照该边界面来投影
            orign_TF = np.zeros((tdim,tdim),dtype=float)
            for i in range(len(Correct_edge_index)):
                cross_edge = Correct_edge_index[i][0] #该边
                idx, = Correct_edge_index[i][1] #对应边界面
                cross_n = n[idx]
                cross_t = t[cross_edge]
                frame_temp = np.zeros((gdim,gdim),dtype=float)
                frame_temp[0] = cross_t
                frame_temp[1] = cross_n
                frame_temp[2] = np.cross(cross_t,cross_n)

                cor_TF = self.Change_frame(frame_temp)
                orign_TF[0] = cor_TF[0]
                base_idx = base1+cross_edge*(tdim-1)*edof
                orign_TF[1:] = self.Tensor_Frame[base_idx:base_idx+(tdim-1)]
                Correct_P = self.Transition_matrix(cor_TF,orign_TF)
                
                self.Change_basis_frame_matrix_edge(M,B0,B1,B2,F0,Correct_P[1:,1:],cross_edge)
                for i_temp in range(edof):
                    #print(base_idx)
                    self.Tensor_Frame[base_idx:base_idx+(tdim-1)] = cor_TF[1:] #对原始标架做矫正
                    base_idx+=(tdim-1)

                #固定自由度赋值
                frame_temp = frame_temp[[1,2,0]]#(3,gdim)
                idx_temp, = np.nonzero(bd_face2edge[idx]==cross_edge)[0]
                val_temp = bd_val[idx,idx_temp] #(edof,gdim)
                #print(val_temp)
                val_temp = np.einsum('ij,kj->ki',val_temp,frame_temp)#(3,edof)
                #print(val_temp)
                bdedge_index_type = bd_index_type[:,idx]#(3,)
                #base_idx = base1+cross_edge*(tdim-1)*edof
                #bd_dof_temp = base_idx+(np.arange(edof)*(tdim-1))+np.array([0,2,4],dtype=int)[:,None] #(3,edof)
                bd_dof_temp = bd_faceedge2dof[idx,idx_temp] #(edof,tdim-1)
                bd_dof_temp = bd_dof_temp[:,[0,2,4]]#(edof,3)
                #print(bd_dof_temp)
                #print(bd_dof_temp,cross_edge)

                if bdedge_index_type[0]:
                    uh[bd_dof_temp[:,0]] = val_temp[0]
                    isBdDof[bd_dof_temp[:,0]] = True
                
                if bdedge_index_type[1]:
                    uh[bd_dof_temp[:,1]] = np.sqrt(2.0)*val_temp[1]
                    isBdDof[bd_dof_temp[:,1]] = True
                
                if bdedge_index_type[2]:
                    uh[bd_dof_temp[:,2]] = np.sqrt(2.0)*val_temp[2]
                    isBdDof[bd_dof_temp[:,2]] = True

            #######################################
            #此时为相交边，每条边有两个边界面
            frame_temp = np.zeros((gdim,gdim),dtype=float)
            orign_TF = np.zeros((tdim,tdim),dtype=float)
            if len(Cross_edge_index)>0:
                #此时需要特殊选取标架
                for i in range(len(Cross_edge_index)):
                    cross_edge = Cross_edge_index[i][0] #该边
                    idx = Cross_edge_index[i][1] #对应边界面
                    cross_n = n[idx] #(2,gdim)
                    cross_t = t[cross_edge] #(gdim,)

                    bdedge_index_type = bd_index_type[:,idx] #(3,2)
                    Ncorner_bdcro_type= np.sum(bdedge_index_type,axis=0) #约束个数

                    frame_temp[0] = cross_t

                    if np.sum(Ncorner_bdcro_type) == 6:
                        #有六个约束，选取一个方向来重新取标架
                        frame_temp[1] = cross_n[0]
                        frame_temp[2] = np.cross(frame_temp[0],frame_temp[1])

                        cor_TF = self.Change_frame(frame_temp)
                        orign_TF[0] = cor_TF[0]
                        base_idx = base1+cross_edge*(tdim-1)*edof
                        orign_TF[1:] = self.Tensor_Frame[base_idx:base_idx+(tdim-1)]
                        Correct_P = self.Transition_matrix(cor_TF,orign_TF)
                        #print(Correct_P)

                        self.Change_basis_frame_matrix_edge(M,B0,B1,B2,F0,Correct_P[1:,1:],cross_edge)
                        for i_temp in range(edof):
                            self.Tensor_Frame[base_idx:base_idx+(tdim-1)] = cor_TF[1:] #对原始标架做矫正
                            base_idx+=(tdim-1)


                        #固定自由度赋值,先做主方向, 再做辅助方向
                        ############################主方向
                        frame_temp = frame_temp[[1,2,0]]# [n,tn,t]
                        idx_temp, = np.nonzero(bd_face2edge[idx[0]]==cross_edge)[0]
                        val_temp = bd_val[idx[0],idx_temp] #(edof,gdim)
                        #print(val_temp.shape)
                        val_temp = np.einsum('ij,kj->ik',val_temp,frame_temp)#(edof,3)
                        val_temp[:,1:] = np.sqrt(2.0)*val_temp[:,1:]
                        bd_dof_temp = bd_faceedge2dof[idx[0],idx_temp] #(edof,tdim-1)
                        #print(bd_dof_temp.shape)
                        bd_dof_temp = bd_dof_temp[:,[0,2,4]]#(edof,3), 

                        uh[bd_dof_temp] = val_temp
                        isBdDof[bd_dof_temp] = True

                        
                        ############################辅方向
                        frame_times_n1 = np.einsum('ij,jkl,l->ik',cor_TF[1:],self.T,cross_n[1]) #(tdim-1,gdim)

                        #frame_temp[1] = cross_n[1]
                        #frame_temp[2] = np.cross(frame_temp[0],frame_temp[1])

                        idx_temp_1, = np.nonzero(bd_face2edge[idx[1]]==cross_edge)[0]
                        val_temp_1 = bd_val[idx[1],idx_temp_1] #(edof,gdim)
                        #print(val_temp_1[0])
                        #print(val_temp[0])

                        val_temp_1 = val_temp_1 - np.einsum('ij,jk->ik',val_temp,frame_times_n1[[0,2,4]])#(edof,gdim)

                        bd_dof_temp = bd_faceedge2dof[idx[1],idx_temp_1] #(edof,tdim-1)
                        bd_dof_temp = bd_dof_temp[:,[1,3]]#(edof,2)

                        uh[bd_dof_temp[:,1]] = np.sqrt(2.0)*np.einsum('ij,j->i',val_temp_1,cross_t)/np.dot(frame_temp[1],cross_n[1])
                        uh[bd_dof_temp[:,0]] = np.einsum('ij,j->i',val_temp_1,frame_temp[1])/np.dot(frame_temp[1],cross_n[1])

                        isBdDof[bd_dof_temp] = True


                    else:
                        print('TODO NEXT')
                        kill
                        


                        





    
        

    def set_essential_bc_vertex(self,index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,B2,F0):
        NFbd = len(index)
        tdim = self.tensor_dimension()
        gdim = self.geo_dimension()
        node_dof_flag, = np.nonzero(self.face_dof_falgs_1()[0])
        #print(node_dof_flag)
        bd_val = val[:,node_dof_flag]#(NFbd,3,gdim)
        #print(bd_val[0,0])
        bdnode2dof = facebd2dof[:,node_dof_flag] #(NFbd,3,tdim)
        bd_face2node = np.array(bdnode2dof[:,:,0]/tdim,dtype=int)#(NFbd,3)
        #print(bd_face2node[0])
        bdnode = np.unique(bd_face2node)
        #print(bdnode)
        Corner_point_index_all = np.array(self.Corner_point_index,dtype=int)
        #print(bdnode,Corner_point_index_all)
        bdnode = np.setdiff1d(bdnode,Corner_point_index_all)
        INNbd = len(bdnode) #边界非角点个数
        node2face_idx = np.zeros(INNbd,dtype=int) #(INNbd,)
        #print(bdnode)
        #node = self.mesh.entity('node')
        #print(node[bdnode])
        #####################################
        #边界内部顶点插值
        if INNbd>0:
            #print(INNbd)
            bd_face2node = bd_face2node.T.reshape(-1)
            idx = (bdnode[:,None]==bd_face2node)
            #print(bdnode,idx)
            bd_face2node = bd_face2node.reshape(3,NFbd).T #(NFbd,3)
            for i in range(INNbd):
                 idx_temp = np.nonzero(idx[i])[0]
                 node2face_idx[i] = idx_temp[0]
            node2face_idx_idx = node2face_idx//NFbd #表示在对应面中的顺序
            node2face_idx = np.mod(node2face_idx,NFbd)
            #print(node2face_idx[1],node2face_idx_idx[1])
            bd_dof = bdnode2dof[node2face_idx,node2face_idx_idx]#(INNbd,tdim) 固定边界点的自由度
            bd_dof = bd_dof[:,[0,5,4]]
            #print(bd_dof[1])
            bdnode_index_type = bd_index_type[:,node2face_idx] #(3,INNbd)
            val_temp = bd_val[node2face_idx,node2face_idx_idx] #(INNbd,gdim)
            #print(val_temp[1])
            frame_temp = frame[node2face_idx] #(INNbd,3,gdim)
            #print(val_temp[[0]])
            #print(node[bdnode][[0]],bdnode[0])
            #print(frame_temp[0],node2face_idx[0])
            #print(bdnode[0],node2face_idx[0],index[16])
            #print(self.Tensor_Frame[bd_dof[0]])
            val_temp = np.einsum('ij,ikj->ik',val_temp,frame_temp)#(INNbd,3)
            bdTensor_Frame = self.Tensor_Frame[bd_dof]#(INNbd,3,tdim)
            T = self.T #(tdim,gidm,gdim)
            bdTFn = np.einsum('ijk,klm,il,ijm->ij',bdTensor_Frame,T,frame_temp[:,0],frame_temp)
            val_temp = val_temp/bdTFn
            for i in range(3):
                idx, = np.nonzero(bdnode_index_type[i])
                if len(idx)>0:
                    uh[bd_dof[idx,i]] = val_temp[:,i][idx]
                    isBdDof[bd_dof[idx,i]] = True


        
        #######################################
        #边界角点插值，角点特殊处理,数量较少，逐点处理即可
        #预判段角点
        
        Corner_point_to_face_index_all = self.Corner_point_bdFace_index
        Correct_point_index = []
        Corner_point_index = []
        for i in range(len(Corner_point_index_all)):
            corner_point = Corner_point_index_all[i]
            corner_point_to_face_index = np.array(Corner_point_to_face_index_all[i],dtype=int)# 一个1维数组，可能有多个分量
            idx, = np.nonzero(np.sum(index[:,None] == corner_point_to_face_index,axis=-1)) #查看该边界边是否是nuemann边界
            if len(idx) == 1: #此时只有一面是Neumann边界，需要变换标架，按照该边界边来投影
                Correct_point_index.append([corner_point,idx])
            elif len(idx) > 1:
                Corner_point_index.append([corner_point,idx])

        
        ##此时只有一边是Neumann边界，需要变换标架，按照该边界边来投影
        for i in range(len(Correct_point_index)):
            corner_point = Correct_point_index[i][0] #该角点
            idx, = Correct_point_index[i][1] #对应的边界面
            frame_temp = frame[idx] #(3,gidm)
            #print(frame_temp)

            cor_TF = self.Change_frame(frame_temp)
            orign_TF = self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim]
            Correct_P = self.Transition_matrix(cor_TF,orign_TF)
            self.Change_basis_frame_matrix(M,B0,B1,B2,F0,Correct_P,corner_point)
            self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF #对原始标架做矫正

            #固定自由度赋值
            idx_temp, = np.nonzero(bd_face2node[idx]==corner_point)[0]
            val_temp = bd_val[idx,idx_temp] #(gdim,)
            bdnode_index_type = bd_index_type[:,idx]
            #print(val_temp,frame_temp[0])
            if bdnode_index_type[0]:
                uh[corner_point*tdim] = np.dot(val_temp,frame_temp[0])
                isBdDof[corner_point*tdim] = True
            if bdnode_index_type[1]:
                uh[corner_point*tdim+5] = np.sqrt(2.0)*np.dot(val_temp,frame_temp[1])
                isBdDof[corner_point*tdim+5] = True
            if bdnode_index_type[2]:
                uh[corner_point*tdim+4] = np.sqrt(2.0)*np.dot(val_temp,frame_temp[2])
                isBdDof[corner_point*tdim+4] = True

        ##多个面都是Neumann边界, 要做准插值
        if len(Corner_point_index)>0:
            #原则，满足应力边界条件最多的面，选择其作为标架，余下的尽量满足
            for i in range(len(Corner_point_index)):
                corner_point = Corner_point_index[i][0]
                idx = Corner_point_index[i][1] #对应的边界面
                NnF = len(idx) #相关面的个数
                frame_temp = frame[idx] #(NnF,3,gidm)


                bdnode_index_type = bd_index_type[:,idx] #(3,NnF)
                Num_bdnone_idx = np.sum(bdnode_index_type,axis=0)

                if np.max(Num_bdnone_idx)==3:
                    #找标架
                    i_temp = 0
                    j_temp = 0
                    while i_temp == 0:
                        if Num_bdnone_idx[j_temp] == 3:
                            frame_n = frame_temp[j_temp] #(3,gdim)

                            cor_TF = self.Change_frame(frame_n)
                            orign_TF = self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim]
                            #print(cor_TF,orign_TF)
                            Correct_P = self.Transition_matrix(cor_TF,orign_TF)
                            self.Change_basis_frame_matrix(M,B0,B1,B2,F0,Correct_P,corner_point)
                            self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF #对原始标架做矫正


                            #固定自由度赋值
                            idx_temp, = np.nonzero(bd_face2node[idx[j_temp]]==corner_point)[0]
                            val_n = bd_val[idx[j_temp],idx_temp] #(gdim,)
                            #print(val_n)
                            val_n = np.einsum('j,ij->i',val_n,frame_n)#(3,)
                            val_n[1:] = np.sqrt(2.0)*val_n[1:]

                            uh[corner_point*tdim] = val_n[0]
                            isBdDof[corner_point*tdim] = True

                            uh[corner_point*tdim+5] = val_n[1]
                            isBdDof[corner_point*tdim+5] = True

                            uh[corner_point*tdim+4] = val_n[2]
                            isBdDof[corner_point*tdim+4] = True



                            frame_temp = np.delete(frame_temp,j_temp,axis=0)
                            bdnode_index_type = np.delete(bdnode_index_type,j_temp,axis=1)
                            Num_bdnone_idx = np.delete(Num_bdnone_idx,j_temp,axis=0)
                            idx = np.delete(idx,j_temp,axis=0)
                            NnF = NnF-1
                            i_temp = 1
                        else:
                            j_temp+=1



                    #还有1，2，3三个值不知道
                    know_idx = np.array([0,5,4],dtype=int)
                    uknow_idx = np.arange(3)+1 #剩余自由度没有确定的
                    A_temp = []
                    b_temp = []

                    for j_temp in range(NnF):
                        idx_temp, = np.nonzero(bd_face2node[idx[j_temp]]==corner_point)[0]
                        val_temp = bd_val[idx[j_temp],idx_temp] #(gdim,)
                        #if NnF == 2:
                            #print(val_temp)
                            #print(cor_TF)
                            #print(frame_n)
                            #print(val_n)
                        frame_temp_temp = frame_temp[j_temp] #(3,gdim)
                        n_temp = frame_temp_temp[0]#(gdim,)

                        frame_n_times_n1 = np.einsum('ij,jkl,l->ik',cor_TF,self.T,n_temp)#(tdim,gdim)
                        val_temp = val_temp - np.einsum('j,jk->k',val_n,frame_n_times_n1[know_idx])#(gdim,)
                        frame_n_times_n1 = frame_n_times_n1[uknow_idx]#(3,gdim)

                        bdnode_index_type_temp = bdnode_index_type[:,j_temp]
                        
                        for i_temp in range(3):
                            if bdnode_index_type_temp[i_temp]:
                                A_temp_temp = np.einsum('ij,j->i',frame_n_times_n1,frame_temp_temp[i_temp])
                                b_temp_temp = np.einsum('j,j->',val_temp,frame_temp_temp[i_temp])
                                if np.max(np.abs(A_temp_temp))>1e-13:
                                    A_temp.append(A_temp_temp)#(3,)
                                    b_temp.append(b_temp_temp)
                                elif np.abs(b_temp_temp)>1e-13:
                                    ValueError('不相容条件，目前不能处理')

                    A_temp = np.array(A_temp,dtype=float)
                    b_temp = np.array(b_temp,dtype=float)
                    

                    if len(b_temp)>=3:
                        #print(A_temp)
                        #print(b_temp)
                        b_temp = np.einsum('ij,i->j',A_temp,b_temp)
                        A_temp = np.einsum('ij,ik->jk',A_temp,A_temp)
                        uh[corner_point*tdim+uknow_idx] = np.linalg.solve(A_temp,b_temp)
                        isBdDof[corner_point*tdim+uknow_idx] = True
                        #print(np.linalg.solve(A_temp,b_temp),uknow_idx)
                    elif len(b_temp) == 2:
                        U,Lam,V = np.linalg.svd(A_temp)
                        b_temp = np.einsum('ij,i->j',U,b_temp)/Lam
                        b_temp = np.einsum('ij,i->j',V[:len(b_temp)],b_temp)
                        idx_temp = np.array(np.nonzero(np.abs(V[2])<1e-13))

                        uh[corner_point*tdim+1+idx_temp] = b_temp[idx_temp]
                        isBdDof[corner_point*tdim+1+idx_temp] = True
                    elif len(b_temp) == 1:
                        idx_temp = np.nonzero(np.abs(A_temp)>1e-13)

                        uh[corner_point*tdim+1+idx_temp] = b_temp/A_temp[idx_temp]
                        isBdDof[corner_point*tdim+1+idx_temp] = True


                else:
                    ValueError('TODO NEXT')
                



        #print(bd_face2node,self.mesh.number_of_nodes())









    def set_nature_bc(self, gD, threshold=None, q=None):
        """
        设置 natural边界条件到右端项中，由于是混合元，故此时为u的gD自由边界条件
        若对应应力边界未设置，则默认该边界方向u为0
        """
        mesh = self.mesh
        gdim = self.geo_dimension()
        gdof = self.number_of_global_dofs()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face',index=index)
                flag = threshold(bc)
                flag = (np.sum(flag,axis=0)>0)
                index = index[flag]

        bd2dof = self.face_to_dof()[index] #(NFbd,ldof,tdim)
        n = mesh.face_unit_normal(index=index) #(NFbd,gdim)
        measure = mesh.entity_measure('face',index=index)
        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.face_basis(bcs)[:,index,...] #(NQ,NFbd,ldof,tdim,tdim)
        shape = list(phi.shape)
        shape[-1] = gdim
        phin = np.zeros(shape,dtype=np.float) #sigam*n, (NQ,NFbd,ldof,tdim,gdim)
        phin[...,0] = np.einsum('...ijlk,ik->...ijl',phi[...,[0,5,4]],n)
        phin[...,1] = np.einsum('...ijlk,ik->...ijl',phi[...,[5,1,3]],n)
        phin[...,2] = np.einsum('...ijlk,ik->...ijl',phi[...,[4,3,2]],n)

        pp = mesh.bc_to_point(bcs,etype='face',index=index)
        _, _, frame = np.linalg.svd(n[:, np.newaxis, :]) # get the axis frame on the face by svd
        t0 = frame[:,1]
        t1 = frame[:,2]
        val = gD(pp,n=n,t0=t0,t1=t1) #(NQ,NFbd,gdim) 此时gD函数,可能给法向分量，也可能给切向分量，具体形式在gD中体现
        bb = np.einsum('m,mil,mijkl,i->ijk', ws, val, phin, measure) #(NFbd,ldof,tdim)
        F = np.zeros(gdof,dtype=np.float)
        idx = bd2dof>=0 #标记出连续边界
        np.add.at(F,bd2dof[idx],bb[idx])
        return F

    


















    def Change_frame(self,frame):
        '''
        3D case
        得到 frame[0]*frame[0].T,frame[1]*frame[1].T, frame[2]*frame[2].T
        (frame[1]*frame[2].T+frame[2]*frame[1].T)/sqrt(2) 
        (frame[0]*frame[2].T+frame[2]*frame[0].T)/sqrt(2) 
        (frame[0]*frame[1].T+frame[1]*frame[0].T)/sqrt(2) 
        
        的向量表示
        '''
        idxx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
        if np.max(np.abs(np.matmul(frame.T,frame)-np.eye(3)))>1e-14:
            print('向量不正交，不能形成正交基')
        else:
            tdim = self.tensor_dimension()
            gdim = self.geo_dimension()
            cor_TF = np.zeros((tdim,tdim),dtype=float)
            for ii, (j, k) in enumerate(idxx):
                cor_TF[ii] = (frame[j, idxx[:, 0]]*frame[k, idxx[:, 1]] + frame[j, idxx[:, 1]]*frame[k, idxx[:, 0]])/2
            cor_TF[gdim:] *=np.sqrt(2)
            return cor_TF

    def Transition_matrix(self,cor_TF,orign_TF):
        '''
        cor_TF = Correct_P*orign_TF
        '''
        gdim = self.geo_dimension()
        orign_TF[:,gdim:] = 2*orign_TF[:,gdim:] #次对角线元乘2,点乘符合矩阵双点乘
        Correct_P = np.einsum('ij,kj->ik',cor_TF,orign_TF)
        orign_TF[:,gdim:] = orign_TF[:,gdim:]/2
        if np.max(np.abs(np.dot(Correct_P,orign_TF)-cor_TF)) < 1e-15:
            return Correct_P
        else:
            print('Some wrong!')
            Kill
            

    def Change_basis_frame_matrix(self,M,B0,B1,B2,F0,Correct_P,corner_point):
        tdim = self.tensor_dimension()      
        #对矩阵A,以及b做矫正
        #print(Correct_P)
        Correct_P = csr_matrix(Correct_P)

        M[corner_point*tdim:corner_point*tdim+tdim] = Correct_P@M[corner_point*tdim:corner_point*tdim+tdim]
        M[:,corner_point*tdim:corner_point*tdim+tdim] = M[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)

        B0[:,corner_point*tdim:corner_point*tdim+tdim] = B0[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)
        B1[:,corner_point*tdim:corner_point*tdim+tdim] = B1[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)
        B2[:,corner_point*tdim:corner_point*tdim+tdim] = B2[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)
        

        F0[corner_point*tdim:corner_point*tdim+tdim] = Correct_P@F0[corner_point*tdim:corner_point*tdim+tdim]


    def Change_basis_frame_matrix_edge(self,M,B0,B1,B2,F0,Correct_P,cross_edge):
        tdim = self.tensor_dimension()    
        NN = self.mesh.number_of_nodes()
        base1 = NN*tdim  
        #对矩阵A,以及b做矫正
        #print(Correct_P)
        Correct_P = csr_matrix(Correct_P) 

        edof = self.edof
        base1+=cross_edge*(tdim-1)*edof 

        for i in range(edof):     

            M[base1:base1+tdim-1] = Correct_P@M[base1:base1+tdim-1]
            M[:,base1:base1+tdim-1] = M[:,base1:base1+tdim-1]@(Correct_P.T)

            B0[:,base1:base1+tdim-1] = B0[:,base1:base1+tdim-1]@(Correct_P.T)
            B1[:,base1:base1+tdim-1] = B1[:,base1:base1+tdim-1]@(Correct_P.T)
            B2[:,base1:base1+tdim-1] = B2[:,base1:base1+tdim-1]@(Correct_P.T)

            F0[base1:base1+tdim-1] = Correct_P@F0[base1:base1+tdim-1]

            base1+=tdim-1









if __name__ == '__main__':
    from fealpy.mesh import MeshFactory
    from fealpy.fem.integral_alg import IntegralAlg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from fealpy.functionspace import LagrangeFiniteElementSpace
    import sys
    import sympy as sp
    from mixed_fem_space import HuZhangFiniteElementSpace as HZspace
    p = int(sys.argv[1])
    mf = MeshFactory()
    mesh = mf.one_triangle_mesh()
    #mesh = mf.one_tetrahedron_mesh()
    mesh.uniform_refine(1)
    space = HuZhangFiniteElementSpace(mesh,p=p)
    spaces = HZspace(mesh,p=p)

    #bc = np.array([[0.2,0.3,0.5],[0.4,0.2,0.4],[1/3,1/3,1/3]],dtype=np.float)
    bc = np.random.random((20,2))
    space.face_basis(bc)

    bc = np.random.random((30,3))
    space.basis(bc)
    space.div_basis(bc)

    gdof = space.number_of_global_dofs()
    uh = np.random.random(gdof)

    space.div_value(uh,bc)


    from linear_elasticity_model import GenLinearElasticitymodel
    x = sp.symbols('x0:2')
    exp = sp.exp
    u = [exp(x[0]-x[1]),exp(x[0]-x[1])]
    pde = GenLinearElasticitymodel(u,x)

    s = pde.stress
    sh = space.interpolation(pde.stress).value
    shs = spaces.interpolation(pde.stress).value


    integrator = mesh.integrator(5)
    measure = mesh.entity_measure()
    integralalg = IntegralAlg(integrator, mesh, measure)
    #print(integralalg.L2_error(s, sh))





    '''


    mesh = mf.one_tetrahedron_mesh()
    mesh.uniform_refine(1)
    space = HuZhangFiniteElementSpace(mesh,p=p)
    spaces = HZspace(mesh,p=p)


    bc = np.random.random((20,2))
    space.edge_basis(bc)

    bc = np.random.random((20,3))
    space.face_basis(bc)

    bc = np.random.random((10,4))
    space.basis(bc)
    space.div_basis(bc)

    gdof = space.number_of_global_dofs()
    uh = np.random.random(gdof)

    space.div_value(uh,bc)


    x = sp.symbols('x0:3')
    exp = sp.exp
    u = [exp(x[0]-x[1]),exp(x[0]-x[1]),exp(x[0]+x[1]+x[2])**2]
    pde = GenLinearElasticitymodel(u,x)

    s = pde.div_stress
    sh = space.interpolation(pde.stress).div_value
    shs = spaces.interpolation(pde.stress).div_value

    integrator = mesh.integrator(5)
    measure = mesh.entity_measure()
    integralalg = IntegralAlg(integrator, mesh, measure)
    #print(integralalg.L2_error(s, shs))
    
    '''
























