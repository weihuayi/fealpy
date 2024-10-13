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
    Hu-Zhang Mixed Finite Element Space 2D.
    """
    def __init__(self, mesh, p, q=None):
        self.space = LagrangeFiniteElementSpace(mesh, p, q=None) # the scalar space
        self.mesh = mesh
        self.p = p
        self.dof = self.space.dof
        self.dim = self.space.GD

        self.edof = (p-1)
        self.fdof = (p-1)*(p-2)//2
        self.cdof = (p-1)*(p-2)//2

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
        self.Tensor_Frame = np.zeros((gdof,tdim),dtype=np.float64) #self.Tensor_Frame[i,:]表示第i个基函数第标架

        
        NE = mesh.number_of_edges()
        idx = np.array([(0, 0), (1, 1), (0, 1)])
        TE = np.zeros((NE, 3, 3), dtype=np.float64)
        self.T = np.array([[(1, 0), (0, 0)], [(0, 0), (0, 1)], [(0, 1), (1, 0)]])

        t = mesh.edge_unit_tangent() 
        _, _, frame = np.linalg.svd(t[:, np.newaxis, :]) # get the axis frame on the edge by svd
        frame[:, 0, :] = t
        for i, (j, k) in enumerate(idx):
            TE[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2
        TE[:, gdim:] *=np.sqrt(2) 


        base0 = 0

        #顶点标架
        T = np.eye(tdim,dtype=np.float64)
        T[gdim:] = T[gdim:]/np.sqrt(2)
        NN = mesh.number_of_nodes()
        shape = (NN,tdim,tdim)
        self.Tensor_Frame[:NN*tdim] = np.broadcast_to(T[None,:,:],shape).reshape(-1,tdim) #顶点标架
        base0 += tdim*NN
        edof = self.edof
        #print(base0)
        if edof > 0: #边内部连续自由度标架
            NE = mesh.number_of_edges()
            shape = (NE,edof,tdim-1,tdim)
            self.Tensor_Frame[base0:base0+NE*edof*(tdim-1)] = np.broadcast_to(TE[:,None,1:],shape).reshape(-1,tdim)
            base0 += NE*edof*(tdim-1)
        #print(base0)
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


        # 对于边界顶点，如果只有一个外法向，选取该外法向来做标架
        bdNode_index = mesh.ds.boundary_node_index() #所有边界顶点点指标
        Nnbd = len(bdNode_index) #边界顶点个数
        bdFace_index = mesh.ds.boundary_face_index() #所有边界面指标
        NFbd = len(bdFace_index) #边界面个数
        bdFace2node = mesh.entity('face')[bdFace_index] #边界面到点的对应 (NFbd,gdim)
        bd_n = mesh.face_unit_normal()[bdFace_index] #边界单位外法向量 #(NFbd,gdim)

        _, _, frame = np.linalg.svd(bd_n[:, np.newaxis, :]) # get the axis frame on the face by svd
        #2D case 保证取法于边标架取法一致
        idx = np.array([(0, 0), (1, 1), (0, 1)])
        frame = frame[:,[1,0],:]
        frame[:,1,:] = bd_n

        bdTF = np.zeros((NFbd, tdim, tdim), dtype=np.float64)
        for i, (j, k) in enumerate(idx):
            bdTF[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2
        bdTF[:, gdim:] *=np.sqrt(2)


        
        self.Corner_point_index = [] #存储角点，用list来存储，规模不大
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
            elif len(indexs) == 2:
                self.Corner_point_index.append(bdNode_index[i])
                self.Corner_point_bdFace_index.append(bdFace_index[indexs])
            else:
                raise ValueError("Warn: The geometry shape is complex, and there are more than two boundary edges related to the corner！")

        



            
    def __str__(self):
        return "Hu-Zhang mixed finite element space 2D!"

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
            edof = self.edof
            NE = mesh.number_of_edges()
            gdof += (tdim-1)*edof*NE # 边内部连续自由度的个数 
            E = mesh.number_of_edges_of_cells() # 单元边的个数
            gdof += NC*E*edof # 边内部不连续自由度的个数 

        if p > 2:
            fdof = self.fdof # 面内部自由度的个数
            if gdim == 2:
                gdof += tdim*fdof*NC

        return gdof 

    def number_of_local_dofs(self):
        ldof = self.dof.number_of_local_dofs()
        tdim = self.tensor_dimension()
        return ldof*tdim

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
        cell2dof = np.zeros((NC, ldof, tdim), dtype=np.int_) # 每个标量自由度变成 tdim 个自由度
        base0 = 0
        base1 = 0

        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来
        idx, = np.nonzero(dofFlags[0]) # 局部顶点自由度的编号
        cell2dof[:, idx, :] = tdim*c2d[:, idx] + np.arange(tdim)
        base1 += tdim*NN # 这是张量自由度编号的新起点
        base0 += NN # 这是标量编号的新起点

        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        if len(idx) > 0:
            # 0号局部自由度对应的是切向不连续的自由度, 留到后面重新编号
            cell2dof[:, idx, 1:] = base1 + (tdim-1)*(c2d[:, idx] - base0) + np.arange(tdim - 1)
            edof = self.edof
            base1 += (tdim-1)*edof*NE
            base0 += edof*NE

        #print(np.max(cell2dof),base1)

        idx, = np.nonzero(dofFlags[2])
        if len(idx) > 0:           
            cell2dof[:, idx, :] = base1 + tdim*(c2d[:, idx] - base0) + np.arange(tdim)
            cdof = self.cdof
            base1 += tdim*cdof*NC
 
        idx, = np.nonzero(dofFlags[1])
        if len(idx) > 0:
            cell2dof[:, idx, 0] = base1 + np.arange(NC*len(idx)).reshape(NC, len(idx))
            base1+=NC*len(idx)
        #print(np.max(cell2dof),base1) 
        self.cell2dof = cell2dof

        
    def init_face_to_dof(self):
        """
        构建局部自由度到全局自由度的映射矩阵
        Returns
        -------
        face2dof : ndarray with shape (NF, ldof,tdim)
            NF: 单元个数
            ldof: p 次标量空间局部自由度的个数
            tdim: 对称张量的维数
        """
        self.face2dof = np.copy(self.edge_to_dof())

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
            edge2dof = np.zeros((NE,ldof,tdim),dtype=np.int_)-1# 每个标量自由度变成 tdim 个自由度

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
        if dim == 2:
            return isOtherDof, isEdgeDof
        else:
            raise ValueError('`dim` should be 2 !')

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
        if gdim == 2:
            return isPointDof, isEdgeDof0, ~(isPointDof | isEdgeDof0)
        else:
            raise ValueError('`dim` should be 2!')

    def face_dof_falgs(self):
        """
        对标量空间中面上的基函数自由度进行分类，分为：
            点上的自由由度
            边内部的自由度
            面内部的自由度        
        """
        p = self.p
        gdim = self.geo_dimension()
        if gdim == 2:
            return self.edge_dof_falgs()
        else:
            raise ValueError('`dim` should be 2!')

    def face_dof_falgs_1(self):
        """
        对标量空间中面上的基函数自由度进行分类，分为：
            点上的自由由度
            边内部的自由度
            面内部的自由度        
        """
        p = self.p
        gdim = self.geo_dimension()
        if gdim == 2:
            return self.edge_dof_falgs_1()
        else:
            raise ValueError('`dim` should be 2!')

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
        if gdim == 2:
            return self.edge_basis(bc)
        else:
            raise ValueError('`dim` should be 2!')


    @barycentric
    def edge_basis(self,bc):
        phi0 = self.space.face_basis(bc) #(NQ,1,ldof)       
        edge2dof = self.edge2dof #(NC,ldof,tdim)
        phi = np.einsum('nijk,...ni->...nijk',self.Tensor_Frame[edge2dof],phi0) #(NE,ldof,tdim,tdim), (NQ,1,ldof)
        #在不连续标架算出的结果不对，但是不影响，因为其自由度就是定义在单元体上的
        #不连续标架有:边内部第0个标架
        return phi #(NQ,NE,ldof,tdim,tdim)  


    @barycentric
    def basis(self, bc, index=np.s_[:],p=None):
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
        phi0 = self.space.basis(bc,index=index,p=p) #(NQ,1,ldof)
        cell2dof = self.cell2dof[index] #(NC,ldof,tdim)

        phi = np.einsum('nijk,...ni->...nijk',self.Tensor_Frame[cell2dof],phi0) #(NC,ldof,tdim,tdim), (NQ,1,ldof)
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


    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dphi = self.div_basis(bc, index=index) #(NQ,NC,ldof,tdim,gdim)
        cell2dof = self.cell_to_dof()
        uh = uh[cell2dof[index]]
        val = np.einsum('...ijkm, ijk->...im', dphi, uh)
        return val #(NQ,NC,gdim)


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
        d = np.array([1, 1, 2])
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

            d = np.array([1, 1, 2])
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

        gdim == 2
        v= [[phi,0],[0,phi]]

        [[B0],[B1]]

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


        I = np.einsum('ij, k->ijk', vspace.cell_to_dof(), np.ones(tldof,dtype=int))
        J = np.einsum('ij, k->ikj', self.cell_to_dof().reshape(NC,-1), np.ones(vldof,dtype=int))   

        B0 = csr_matrix((B0.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))
        B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(vgdof, tgdof))

        return B0, B1

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

        def f(i):
            s = slice(index[i],index[i+1])
            measure = Tmeasure[s]
            c2d0 = cell2dof0[s]
            c2d1 = cell2dof1[s]

            shape = (len(measure),c2d0.shape[1],c2d1.shape[1]) #（lNC,ldof0,ldof1)
            lNC  = index[i+1]-index[i]
            M0 = np.zeros(shape,measure.dtype)
            M1 = np.zeros(shape,measure.dtype)
            for bc, w in zip(bcs, ws):
                phi0 = basis0(bc,index=s)#(1,vldof)
                phi1 = basis1(bc,index=s).reshape(lNC,-1,gdim)#(lNC, ldof*tdim, gdim)
                M0 += np.einsum('jk,jo,j->jko',phi0, phi1[...,0], w*measure)
                M1 += np.einsum('jk,jo,j->jko',phi0, phi1[...,1], w*measure)

            I = np.broadcast_to(c2d0[:, :, None], shape=M0.shape)
            J = np.broadcast_to(c2d1[:, None, :], shape=M0.shape)

            Bi0 = csr_matrix((M0.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            Bi1 = csr_matrix((M1.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            return Bi0,Bi1

        # 并行组装总矩阵

        with Pool(nc) as p:
            Bi= p.map(f, range(nc))
            
        for val in Bi:
            B0 += val[0]
            B1 += val[1]
        

        return B0, B1



















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

    def array(self, dim=None, dtype=np.float_):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float_)

    def function(self, dim=None):
        f = Function(self)
        return f

    def set_essential_bc(self, uh, gN, M,B0, B1, F0, threshold=None):
        """
        初始化压力的本质边界条件，插值一个边界sigam,使得sigam*n=gN,对于角点，要小心选取标架
        由face2bddof 形状为(NFbd,ldof,tdim)
        2D case 时face2bddof[...,0]--切向标架， face2bddof[...,1]--法向标架， face2bddof[...,2]--切法向组合标架
        """
        mesh = self.mesh
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension()
        ipoint = self.dof.interpolation_points()
        gdof = self.number_of_global_dofs()
        #node = mesh.entity('node')


        

        if type(threshold) is np.ndarray:
            index = threshold #此种情况后面补充
        else:
            if threshold is not None:
                index = mesh.ds.boundary_face_index()
                bc = mesh.entity_barycenter('face',index=index)
                flag = threshold(bc) #(2,gNEbd),第0行表示给的法向投影，第1行分量表示给的切向投影
                flag_idx = (np.sum(flag,axis=0)>0) #(gNFbd,)
                index = index[flag_idx] #(NFbd,)
                NFbd = len(index)

                bd_index_type = np.zeros((2,NFbd),dtype=np.bool_)
                bd_index_type[0] = flag[0][flag_idx] #第0个分量表示给的法向投影
                bd_index_type[1] = flag[1][flag_idx] #第1个分量表示给的切向投影
                #print(bd_index_type)


        n = mesh.face_unit_normal()[index] #(NEbd,gdim)
        t = mesh.edge_unit_tangent()[index] #(NEbd,gdim)
        isBdDof = np.zeros(gdof,dtype=np.bool_)#判断是否为固定顶点
        Is_cor_face_idx = np.zeros(NFbd,dtype=np.bool_) #含角点的边界边
        f2dbd = self.dof.face_to_dof()[index] #(NEbd,ldof)
        ipoint = ipoint[f2dbd] #(NEbd,ldof,gdim)
        facebd2dof = self.face2dof[index] #(NEbd,ldof,tdim)
        #print(f2dbd,index.shape,facebd2dof.shape)
        frame = np.zeros((NFbd,2,2),dtype=np.float_)
        frame[:,0] = n
        frame[:,1] = t

        val = gN(ipoint,n[...,None,:],t=t[...,None,:]) #(NEbd,ldof,gdim)，可能是法向，也可能是切向，或者两者的线性组合

        #将边界边内部点与顶点分别处理


 

        self.set_essential_bc_inner_edge(facebd2dof,bd_index_type,frame,val,uh,isBdDof)#处理所有边界边内部点
        self.set_essential_bc_vertex(index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,F0)#处理所有边界边顶点

        return isBdDof

    def set_essential_bc_inner_edge(self,facebd2dof,bd_index_type,frame,val,uh,isBdDof):
        #处理所有边界边内部点
        inner_edge_dof_flag, = np.nonzero(self.face_dof_falgs_1()[1])
        val_temp = val[:,inner_edge_dof_flag] #(NFbd,edof,gdim)
        bdinedge2dof = facebd2dof[:,inner_edge_dof_flag,1:] #(NFbd,edof,tdim-1)
        bdTensor_Frame = self.Tensor_Frame[bdinedge2dof] #(NFbd,edof,2,tdim)
        n = frame[:,0]
        for i in range(2):
            bd_index_temp, = np.nonzero(bd_index_type[i])
            if len(bd_index_temp)>0:
                bdTensor_Frame_projection = np.einsum('ijl,lmn,in,im->ij',bdTensor_Frame[bd_index_temp,:,i,:],
                                                               self.T,frame[bd_index_temp,i],n[bd_index_temp])
                val_projection = np.einsum('ijk,ik->ij',val_temp[bd_index_temp],frame[bd_index_temp,i])
                uh[bdinedge2dof[bd_index_temp,:,i]] = val_projection/bdTensor_Frame_projection
                isBdDof[bdinedge2dof[bd_index_temp,:,i]] = True
                #print(uh[bdinedge2dof[bd_index_temp,:,i]])

    def set_essential_bc_vertex(self,index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,F0):
        NFbd = len(index)
        tdim = self.tensor_dimension()
        gdim = self.geo_dimension()
        node_dof_flag, = np.nonzero(self.face_dof_falgs_1()[0])
        bd_val = val[:,node_dof_flag] #(NFbd,2,gdim)
        bdnode2dof = facebd2dof[:,node_dof_flag] #(NFbd,2,tdim)
        #print(facebd2dof[:,:,0]/tdim)
        bd_edge2node = np.array(bdnode2dof[:,:,0]/tdim,dtype=int)#(NFbd,2)
        bdnode = np.unique(bd_edge2node)#boundary vert index
        Corner_point_index_all = np.array(self.Corner_point_index,dtype=int) #所有边界点
        bdnode = np.setdiff1d(bdnode,Corner_point_index_all)
        INNbd = len(bdnode) #边界非角点个数
        node2edge_idx = np.zeros((INNbd,2),dtype=int) #(INNbd,2)

        #####################################
        #边界内部顶点插值
        #print(bd_edge2node)
        bd_edge2node = bd_edge2node.T.reshape(-1)
        idx = bdnode[:,None]==bd_edge2node
        node2edge_idx[:,0] = np.argwhere(idx[:,:NFbd])[:,1] 
        node2edge_idx[:,1] = np.argwhere(idx[:,NFbd:])[:,1]
        bd_dof = bdnode2dof[node2edge_idx[:,0],0,1:] #(INNbd,2) 固定边界点的自由度
        bdnode_index_type = bd_index_type[:,node2edge_idx[:,0]] #(2,INNbd) 边界自由度类型
        val_temp = bd_val[node2edge_idx,[0,1]] #(INNbd,2,gdim)
        val_temp = np.einsum('ijk,ijlk->lji',val_temp,frame[node2edge_idx]) #(2,2,INNbd,)

        for i in range(2):
            idx, = np.nonzero(bdnode_index_type[i])
            if len(idx)>0:
                if i == 0:   
                    uh[bd_dof[idx,i]] = np.sum(val_temp[i][:,idx],axis=0)/2.0  
                else:
                    Tensor_Frame = self.Tensor_Frame[bd_dof[idx,1]] #(True_INNbd,tdim) #可能差个负号，引入
                    n_temp = frame[node2edge_idx[idx]][:,:,0] #(INNbd,2,gdim)
                    t_temp = frame[node2edge_idx[idx]][:,:,1] #(INNbd,2,gdim)
                    Tnt = np.einsum('lk,kij,lsi,lsj->sl',Tensor_Frame,self.T,n_temp,t_temp) #(2,INNbd)
                    val_temp = val_temp[i][:,idx] #(2,INNbd)
                    uh[bd_dof[idx,i]] = np.einsum('ij,ij->j',val_temp,Tnt)/np.einsum('ij,ij->j',Tnt,Tnt)
    

                isBdDof[bd_dof[idx,i]] = True
        #######################################
        #边界角点插值，角点特殊处理,数量较少，逐点处理即可
        #预判段角点
        n = frame[:,0]
        t = frame[:,1]
        Corner_point_to_face_index_all = np.array(self.Corner_point_bdFace_index,dtype=int) 
        Correct_point_index = []
        Corner_point_index = []
        #Total_Corner_point = [] 
        for i in range(len(Corner_point_index_all)):
            corner_point = Corner_point_index_all[i]
            corner_point_to_face_index = Corner_point_to_face_index_all[i]# 一个1维数组，2D情况下只有两个分量
            idx, = np.nonzero(np.sum(index[:,None] == corner_point_to_face_index,axis=-1)) #查看该边界边是否是nuemann边界
            if len(idx) == 1: #此时只有一边是Neumann边界，需要变换标架，按照该边界边来投影
                Correct_point_index.append([corner_point,idx])
            elif len(idx) == 2:
                Corner_point_index.append([corner_point,idx])
                #Total_Corner_point.append(corner_point)


        bd_edge2node = bd_edge2node.reshape(2,NFbd).T #(NFbd,2)

        ##此时只有一边是Neumann边界，需要变换标架，按照该边界边来投影
        for i in range(len(Correct_point_index)):
            corner_point = Correct_point_index[i][0] #该角点
            idx, = Correct_point_index[i][1] #对应的边界边
            
            corner_n = n[idx]
            corner_t = t[idx]
            frame = np.zeros((gdim,gdim),dtype=float)
            frame[1] = corner_n
            frame[0] = corner_t


            cor_TF = self.Change_frame(frame)
            orign_TF = self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
            Correct_P = self.Transition_matrix(cor_TF,orign_TF)
            self.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
            self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF #对原始标架做矫正

            #固定自由度赋值
            idx_temp, = np.nonzero(bd_edge2node[idx] == corner_point)[0]
            val_temp = bd_val[idx,idx_temp] #(gdim,)
            bdnode_index_type = bd_index_type[:,idx]
            if bdnode_index_type[0]:
                uh[corner_point*tdim+1] = np.dot(val_temp,corner_n)
                isBdDof[corner_point*tdim+1] = True
            if bdnode_index_type[1]:
                uh[corner_point*tdim+2] = np.sqrt(2.0)*np.dot(val_temp,corner_t)
                isBdDof[corner_point*tdim+2] = True

  
        ##两边都是Neumann边界, 要做准插值
        T = np.eye(tdim,dtype=np.float_)
        T[gdim:] = T[gdim:]/np.sqrt(2)
        for i in range(len(Corner_point_index)):
            corner_point = Corner_point_index[i][0] #该角点
            idx = Corner_point_index[i][1] #该角点对应的边
            corner_n = n[idx] #找到角点对应的两个法向 (2,gdim)
            corner_t = t[idx] #找到角点对应点两个切向 (2,gdim)
            corner_index_type = bd_index_type[...,idx] #分量类型判断(2,2)
            Ncorner_bdfix_type = np.sum(corner_index_type,axis=0)

            idx_temp = np.argwhere(bd_edge2node[idx] == corner_point)[:,1]
            val_temp = bd_val[idx,idx_temp] #(2,gdim)

            val_temp_n = np.einsum('ij,ij->i',val_temp,corner_n)#(2,)
            val_temp_t = np.einsum('ij,ij->i',val_temp,corner_t)#(2,)

            #按照约束个数来判断类型
            if np.sum(Ncorner_bdfix_type)==4:
                #有四个约束，足够确定三个自由度，最小二乘方法来确定即可
                Tn = np.einsum('ij,jkl,ml->mik',T,self.T,corner_n) #(2,tdim,gdim)
                Tnn = np.einsum('mik,mk->mi',Tn,corner_n) #(2,tdim)
                Tnt = np.einsum('mik,mk->mi',Tn,corner_t) #(2,tdim)

                A_temp = np.array([Tnn,Tnt]).reshape(-1,tdim)
                b_temp = np.array([val_temp_n,val_temp_t]).reshape(-1)

                b_temp = np.einsum('ij,i->j',A_temp,b_temp)
                A_temp = np.einsum('ik,il->kl',A_temp,A_temp)
                uh[corner_point*tdim:corner_point*tdim+tdim] = np.linalg.solve(A_temp,b_temp)
                isBdDof[corner_point*tdim:corner_point*tdim+tdim] = True

                #表明有四个约束，最小二乘直接求解
            elif np.sum(Ncorner_bdfix_type)==3:
                #有三个约束，刚好可能确定三个自由度，
                #而且有一条边两个标架都有，考虑变换标架为该条边方向
                frame_all = np.zeros((2,2,gdim),dtype=np.float) #按边顺序放法向，切向向量
                frame_all[:,0,:] = corner_n
                frame_all[:,1,:] = corner_t
                if Ncorner_bdfix_type[0]==2:
                    #第0条边的切，法向分量来构建标架
                    frame_idx = 0
                    temp_idx = 1
                elif Ncorner_bdfix_type[1]==2:
                    #第1条边的切，法向分量来构建标架
                    frame_idx = 1
                    temp_idx = 0
                
                frame_edge = frame_all[frame_idx]
                frame_temp = frame_all[temp_idx]
                cor_TF = self.Change_frame(frame_edge[[1,0]])
                orign_TF = self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
                Correct_P = self.Transition_matrix(cor_TF,orign_TF)
                self.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
                self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF

                uh_pre = np.zeros(3) #可能为三个，也可能只有两个
                uh_pre[1] = val_temp_n[frame_idx]
                uh_pre[2] = np.sqrt(2.0)*val_temp_t[frame_idx]

                if corner_index_type[0,temp_idx]:
                    #表明是法向
                    uh_pre[0] = val_temp_n[temp_idx]
                    A_temp=np.einsum('ij,jkl,k,l->i',cor_TF,self.T,corner_n[temp_idx],corner_n[temp_idx])
                elif corner_index_type[1,temp_idx]:
                    #表明是切向
                    uh_pre[0] = val_temp_t[temp_idx]
                    A_temp=np.einsum('ij,jkl,k,l->i',cor_TF,self.T,corner_n[temp_idx],corner_t[temp_idx])
                uh_pre[0] -= (uh_pre[1]*A_temp[1]+uh_pre[2]*A_temp[2])
                #print(np.abs(uh_pre[0])/np.max(np.abs(uh_pre)))
                if np.abs(A_temp[0])>1e-15:
                    uh_pre[0] = uh_pre[0]/A_temp[0]
                    uh[corner_point*tdim:corner_point*tdim+tdim] = uh_pre
                    isBdDof[corner_point*tdim:corner_point*tdim+tdim] = True
                elif np.abs(uh_pre[0])/np.max(np.abs(uh_pre)) < 1e-14:
                    uh[corner_point*tdim+1:corner_point*tdim+tdim] = uh_pre[1:]
                    isBdDof[corner_point*tdim+1:corner_point*tdim+tdim] = True
                else:
                    raise ValueError('角点赋值不相容')

            elif np.sum(Ncorner_bdfix_type)==2:
                #有两个约束，确定两个方向，要特殊选取标架
                frame_all = np.zeros((2,2,gdim),dtype=np.float) #按边顺序放法向，切向向量
                frame_all[:,0,:] = corner_n
                frame_all[:,1,:] = corner_t
                idx_temp = np.zeros(2,dtype=np.int)
                idx_temp[0], = np.nonzero(corner_index_type[:,0])
                idx_temp[1], = np.nonzero(corner_index_type[:,1])
                corner_projection = frame_all[[0,1],idx_temp]

                orign_TF = self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
                orign_matrix = np.einsum('lk,kij,mj,mi->ml',orign_TF,self.T,corner_n,corner_projection) #(2,tdim)
                U,Lam,Correct_P = np.linalg.svd(orign_matrix)

                cor_TF = np.einsum('ik,kj->ij',Correct_P,orign_TF) #(tdim,tdim)
                self.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
                self.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF

                corner_gNb = np.einsum('ij,ij->i',val_temp,corner_projection)
                corner_gNb = np.einsum('ij,i->j',U,corner_gNb)


                if np.abs(Lam[1])>1e-15:
                        uh[corner_point*tdim:corner_point*tdim+tdim-1] = corner_gNb/Lam
                        isBdDof[corner_point*tdim:corner_point*tdim+tdim-1] = True
                else:
                    if np.abs(corner_gNb[1])>1e-15:
                        raise ValueError('角点赋值不相容')
                    uh[corner_point*tdim] = corner_gNb[0]/Lam[0]
                    isBdDof[corner_point*tdim] = True







  
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
        
        bd2dof = self.edge_to_dof()[index] #(NEbd,ldof,tdim)

       
        n = mesh.face_unit_normal(index=index) #(NFbd,gdim)
        measure = mesh.entity_measure('face',index=index)
        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.face_basis(bcs)[:,index,...] #(NQ,NFbd,ldof,tdim,tdim)
        shape = list(phi.shape)
        shape[-1] = gdim
        phin = np.zeros(shape,dtype=np.float_) #sigam*n, (NQ,NFbd,ldof,tdim,gdim)

        phin[...,0] = np.einsum('...ijlk,ik->...ijl',phi[...,[0,2]],n)
        phin[...,1] = np.einsum('...ijlk,ik->...ijl',phi[...,[2,1]],n)

        pp = mesh.bc_to_point(bcs,index=index)
        t = mesh.edge_unit_tangent(index=index)
        val = gD(pp,n=n,t=t) #(NQ,NFbd,gdim) 此时gD函数,可能给法向分量，也可能给切向分量，具体形式在gD中体现
        bb = np.einsum('m,mil,mijkl,i->ijk', ws, val, phin, measure) #(NFbd,ldof,tdim)
        idx = bd2dof>=0 #标记出连续边界
        F = np.zeros(gdof,dtype=np.float_)
        np.add.at(F,bd2dof[idx],bb[idx])
        return F


    def Change_frame(self,frame):
        '''
        2D case
        得到 frame[0]*frame[0].T,frame[1]*frame[1].T,
        (frame[1]*frame[0].T+frame[0]*frame[1].T)/sqrt(2) 的向量表示
        '''
        idxx = np.array([(0, 0), (1, 1), (0, 1)])
        if np.abs(np.dot(frame[0],frame[1]))>1e-15:
            raise ValueError('两个向量不正交，不能形成正交基')
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
        Correct_P = np.einsum('ij,kj->ik',cor_TF,orign_TF)  #TODO  check it is right
        orign_TF[:,gdim:] = orign_TF[:,gdim:]/2
        #print(np.max(np.abs(np.dot(Correct_P,orign_TF)-cor_TF)))
        if np.max(np.abs(np.dot(Correct_P,orign_TF)-cor_TF)) < 1e-13:
            return Correct_P
        else:
            raise ValueError('标架表示有问题！')

    def Change_basis_frame_matrix(self,M,B0,B1,F0,Correct_P,corner_point):
        tdim = self.tensor_dimension()       
        #对矩阵A,以及b做矫正
        #print(Correct_P)
        Correct_P = csr_matrix(Correct_P)                  

        M[corner_point*tdim:corner_point*tdim+tdim] = Correct_P@M[corner_point*tdim:corner_point*tdim+tdim]
        M[:,corner_point*tdim:corner_point*tdim+tdim] = M[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)

        B0[:,corner_point*tdim:corner_point*tdim+tdim] = B0[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)
        B1[:,corner_point*tdim:corner_point*tdim+tdim] = B1[:,corner_point*tdim:corner_point*tdim+tdim]@(Correct_P.T)

        F0[corner_point*tdim:corner_point*tdim+tdim] = Correct_P@F0[corner_point*tdim:corner_point*tdim+tdim]





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



