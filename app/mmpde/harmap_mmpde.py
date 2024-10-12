from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.mesh import TetrahedronMesh
from fealpy.experimental.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh as TM
from fealpy.mesh import TetrahedronMesh as THM
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import (BilinearForm 
                                     ,ScalarDiffusionIntegrator
                                     ,LinearForm
                                     ,ScalarSourceIntegrator
                                     ,DirichletBC)
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix,spdiags,hstack,block_diag,bmat,diags
from sympy import *
from typing import Union

class LogicMesh():
    def __init__(self , mesh: Union[TriangleMesh,TetrahedronMesh]) -> None:
        """
        mesh : 物理网格
        """
        self.mesh = mesh
        self.TD = mesh.top_dimension()
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')
        self.edge = mesh.entity('edge')

        self.BdNodeidx = mesh.boundary_node_index()
        self.BdFaceidx = mesh.boundary_face_index()
        # 新网格下没有该方法
        if self.TD ==2:
            self.node2edge = TM(self.node, self.cell).ds.node_to_edge()
        else:
            self.node2face = self.node_to_face()
        self.local_n2f_norm,self.normal = self.get_basic_information()

    def get_logic_mesh(self) :
        if self.TD == 2:
            if self.is_convex():
                logic_mesh = TriangleMesh(self.node.copy(),self.cell)
            else:
                logic_node = self.get_logic_node()
                logic_cell = self.cell
                logic_mesh = TriangleMesh(logic_node,logic_cell)
            return logic_mesh
        elif self.TD == 3:
            if not self.is_convex():
                raise ValueError('非凸多面体无法构建逻辑网格')
            else:
                logic_mesh = TetrahedronMesh(self.node.copy(),self.cell)
            return logic_mesh
    
    def is_convex(self):
        """
        判断边界是否是凸的
        """
        from scipy.spatial import ConvexHull
        ln2f_norm = self.local_n2f_norm
        vertex_idx = self.BdNodeidx[ln2f_norm[:,-1] >= 0]
        intnode = self.node[vertex_idx]
        hull = ConvexHull(intnode)
        return len(intnode) == len(hull.vertices)
    
    def node_to_face(self): # 作为三维网格的辅助函数
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NF = mesh.number_of_faces()

        face = mesh.entity('face')
        NVF = 3
        node2face = csr_matrix(
                (
                    bm.ones(NVF*NF, dtype=bm.bool),
                    (
                        face.flat,
                        bm.repeat(range(NF), NVF)
                    )
                ), shape=(NN, NF))
        return node2face
    
    def sort_bdnode_and_bdface(self):
        """
        对二维边界点和边界面进行排序
        """
        BdNodeidx = self.BdNodeidx
        BdEdgeidx = self.BdFaceidx 
        node = self.node
        edge = self.edge
        # 对边界边和点进行排序
        node2edge = self.node2edge
        bdnode2edge = node2edge[BdNodeidx][:,BdEdgeidx]
        i,j = bm.nonzero(bdnode2edge)
        bdnode2edge = j.reshape(-1,2)
        glob_bdnode2edge = bm.zeros_like(node,dtype=bm.int32)
        glob_bdnode2edge = bm.set_at(glob_bdnode2edge,BdNodeidx,BdEdgeidx[bdnode2edge])
        
        sort_glob_bdedge_idx_list = []
        sort_glob_bdnode_idx_list = []

        start_bdnode_idx = BdNodeidx[0]
        sort_glob_bdnode_idx_list.append(start_bdnode_idx)
        current_node_idx = start_bdnode_idx
        
        for i in range(bdnode2edge.shape[0]):
            if edge[glob_bdnode2edge[current_node_idx,0],1] == current_node_idx:
                next_edge_idx = glob_bdnode2edge[current_node_idx,1]
            else:
                next_edge_idx = glob_bdnode2edge[current_node_idx,0]
            sort_glob_bdedge_idx_list.append(next_edge_idx)
            next_node_idx = edge[next_edge_idx,1]
            # 处理空洞区域
            if next_node_idx == start_bdnode_idx:
                if i < bdnode2edge.shape[0] - 1:
                    remian_bdnode_idx = list(set(BdNodeidx)-set(sort_glob_bdnode_idx_list))
                    start_bdnode_idx = remian_bdnode_idx[0]
                    next_node_idx = start_bdnode_idx
                else:
                # 闭环跳出循环
                    break
            sort_glob_bdnode_idx_list.append(next_node_idx)
            current_node_idx = next_node_idx
        return bm.array(sort_glob_bdnode_idx_list,dtype=bm.int32),\
               bm.array(sort_glob_bdedge_idx_list,dtype=bm.int32)
    
    def get_basic_information(self):
        """
        返回逻辑网格的基本信息
        """
        BdNodeidx = self.BdNodeidx
        BdFaceidx = self.BdFaceidx
        if self.TD == 3:
            node2face = self.node_to_face()
        else:
            mesh = TM(self.node,self.cell)
            node2face = mesh.ds.node_to_face()
            BdNodeidx,BdFaceidx = self.sort_bdnode_and_bdface()
            self.BdNodeidx = BdNodeidx

        bd_node2face = node2face[BdNodeidx][:,BdFaceidx]
        i , j = bm.nonzero(bd_node2face)

        bdfun = self.mesh.face_unit_normal(index=BdFaceidx[j])
        normal = bm.unique(bdfun ,axis = 0)
        K = bm.arange(len(normal),dtype = bm.int32)
        _,index = bm.unique(i,return_index = True)
        index = bm.concatenate((index , [j.shape[0]]))

        node2face_normal = -bm.ones((BdNodeidx.shape[0],self.TD),dtype=bm.int32)
        for n in range(BdNodeidx.shape[0]):
            b = bm.arange(index[n],index[n+1])
            lface_normal = bm.unique(bdfun[b],axis = 0)
            num_of_lface_normal =  lface_normal.shape[0]
            if num_of_lface_normal == 1:
                tag = bm.all(normal == lface_normal,axis = 1)@K
                node2face_normal[n,0] = tag
            elif num_of_lface_normal == 2:
                tag = [bm.all(normal == lface_normal[0],axis=1),
                        bm.all(normal == lface_normal[1],axis=1)]@K
                node2face_normal[n,:2] = tag
            else:
                # 无关几个面，只取前三个
                tag = [bm.all(normal == lface_normal[0],axis=1),
                        bm.all(normal == lface_normal[1],axis=1),
                        bm.all(normal == lface_normal[2],axis=1)]@K
                node2face_normal[n,:3] = tag
        return node2face_normal,normal
    
    def get_boundary_condition(self,p) -> TensorLike:
        """
        逻辑网格的边界条件,为边界点的集合
        """
        node = self.node
        local_n2f_norm = self.local_n2f_norm
        Vertexidx = self.BdNodeidx[local_n2f_norm[:,-1] >= 0]
        physics_domain = node[Vertexidx]

        num_sides = physics_domain.shape[0]
        angles = bm.linspace(0,2*bm.pi,num_sides,endpoint=False)

        logic_domain = bm.stack([bm.cos(angles),bm.sin(angles)],axis=1)

        logic_bdnode = bm.zeros_like(node,dtype=bm.float64)
        logic_bdnode = bm.set_at(logic_bdnode,Vertexidx,logic_domain)
        
        Pside_length = bm.linalg.norm(bm.roll(physics_domain,-1,axis=0)
                                - physics_domain,axis=1)
        BDNN = len(self.BdNodeidx)
        K = bm.arange(BDNN,dtype=bm.int32)[local_n2f_norm[:,-1] >= 0]

        for i in range(num_sides):
            G = bm.arange(K[i]+1,K[i+1] if i < num_sides-1 else BDNN)
            side_node = node[self.BdNodeidx[G]]
            side_part_length = bm.linalg.norm(side_node - physics_domain[i]
                                            ,axis=1)
            rate = (side_part_length/Pside_length[i]).reshape(-1,1)

            logic_bdnode = bm.set_at(logic_bdnode,self.BdNodeidx[G],(1-rate)*logic_domain[i]\
                                     + rate*logic_domain[(i+1)%num_sides])

        map = []
        for node_p in p:                                         
            idx = bm.where((node == node_p).all(axis=1))[0][0]
            map.append(idx)
        return logic_bdnode[map]
    
    def get_logic_node(self) -> TensorLike:
        """
        logic_node : 逻辑网格的节点坐标
        """
        mesh = self.mesh
        bdc = self.get_boundary_condition
        p = 1 # 有限元空间次数
        space = LagrangeFESpace(mesh, p=p)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=p+1))
        A = bform.assembly()
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(source=0,q=p+1))
        F = lform.assembly()
        bc0 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,0])
        bc1 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,1])
        uh0 = bm.zeros(space.number_of_global_dofs(),dtype=bm.float64)
        uh1 = bm.zeros(space.number_of_global_dofs(),dtype=bm.float64)
        A1, F1 = bc0.apply(A, F, uh0)
        A2, F2 = bc1.apply(A, F, uh1)
        uh0 = bm.set_at(uh0 , slice(None), spsolve(csr_matrix(A1.toarray()), F1))
        uh1 = bm.set_at(uh1 , slice(None), spsolve(csr_matrix(A2.toarray()), F2))
        logic_node = bm.stack([uh0,uh1],axis=1)
        return logic_node
    

class Harmap_MMPDE(LogicMesh):
    def __init__(self, 
                 mesh:TriangleMesh , 
                 uh:TensorLike,
                 pde ,
                 beta :float ,
                 alpha = 0.5, 
                 mol_times = 1 , 
                 redistribute = True) -> None:
        """
        mesh : 初始物理网格
        uh : 物理网格上的解
        pde : 微分方程基本信息
        beta : 控制函数的参数
        alpha : 移动步长控制参数
        mol_times : 磨光次数
        redistribute : 是否预处理边界节点
        """
        super().__init__(mesh)
        self.uh = uh
        self.pde = pde
        self.beta = beta
        self.alpha = alpha
        self.mol_times = mol_times
 
        self.NN = mesh.number_of_nodes()
        self.BDNN = len(self.BdNodeidx)
        self.cm = mesh.entity_measure('cell')
        # 新网格下没有该方法
        if self.TD == 2:
            self.node2cell = TM(self.node, self.cell).ds.node_to_cell()
        else:
            self.node2cell = THM(self.node, self.cell).ds.node_to_cell()
        self.isBdNode = mesh.boundary_node_flag()
        self.redistribute = redistribute

        self.W = bm.array([[0,1],[-1,0]],dtype=bm.int32)
        self.localEdge = bm.array([[1,2],[2,0],[0,1]],dtype=bm.int32)

        self.logic_mesh = self.get_logic_mesh()
        self.logic_node = self.logic_mesh.entity('node')
        self.isconvex = self.is_convex()

        self.star_measure,self.i,self.j = self.get_star_measure()
        self.G = self.get_control_function(beta,mol_times)
        self.A , self.b = self.get_linear_constraint()

    def get_star_measure(self)->TensorLike:
        """
        计算每个节点的星的测度
        """
        star_measure = bm.zeros(self.NN,dtype=bm.float64)
        i,j = bm.nonzero(self.node2cell)
        bm.add_at(star_measure , i , self.cm[j])
        return star_measure,i,j
    
    def get_control_function(self,beta, mol_times):
        """
        计算控制函数
        beta : 控制函数的参数
        mol_times : 磨光次数
        """
        node = self.node
        cell = self.cell

        cm = self.cm
        gphi = self.mesh.grad_lambda()
        guh_incell = bm.sum(self.uh[self.cell,None] * gphi,axis=1)

        M_incell = bm.sqrt(1  + beta *bm.sum( guh_incell**2,axis=1))
        M = bm.zeros(node.shape[0],dtype=bm.float64)
        if mol_times == 0:
            bm.add_at(M , self.i , (cm *M_incell)[self.j])
        else:
            for k in range(mol_times):
                bm.add_at(M , self.i , (cm *M_incell)[self.j])
                M /= self.star_measure
                M_incell = bm.mean(M[cell],axis=1)
        return 1/M_incell
        
    def get_stiff_matrix(self):
        """
        组装刚度矩阵
        """
        mesh = self.mesh
        q = 3
        qf = mesh.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        space = LagrangeFESpace(mesh, p=1)
        gphi = space.grad_basis(bcs)

        cell2dof = space.cell_to_dof()
        GDOF = space.number_of_global_dofs()
    
        H = bm.einsum('q , cqid , c ,cqjd, c -> cij ',ws, gphi ,self.G , gphi, self.cm)
        I = bm.broadcast_to(cell2dof[:, :, None], shape=H.shape)
        J = bm.broadcast_to(cell2dof[:, None, :], shape=H.shape)
        H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))

        H11 = H[~self.isBdNode][:, ~self.isBdNode]
        H12 = H[~self.isBdNode][:, self.isBdNode]
        H21 = H[self.isBdNode][:, ~self.isBdNode]
        H22 = H[self.isBdNode][:, self.isBdNode]
        return H11,H12,H21,H22
    
    def get_linear_constraint(self):
        """
        组装线性约束
        """
        logic_node = self.logic_node
        NN = self.NN
        BDNN = self.BDNN
        isBdNode = self.isBdNode
        BdNodeidx = self.BdNodeidx
        local_n2f_norm = self.local_n2f_norm
        normal = self.normal
        if self.TD == 2:
            isBdinnernode = local_n2f_norm[:,1] < 0
            isVertex = ~isBdinnernode
            b = bm.zeros(NN,dtype=bm.float64)
            logic_Bdnode = logic_node[BdNodeidx]
            b_val0 = bm.sum(logic_Bdnode*
                            normal[local_n2f_norm[:,0]],axis=1)
            b_val1 = bm.sum(logic_Bdnode[isVertex]*
                            normal[local_n2f_norm[isVertex,1]],axis=1)
            b = bm.set_at(b,BdNodeidx,b_val0)
            b = bm.concatenate([b[isBdNode],b_val1])

            # 处理排序后的边界节点的顺序错位
            A1_diag = bm.zeros(NN  , dtype=bm.float64)
            A2_diag = bm.zeros(NN  , dtype=bm.float64)
            A1_diag = bm.set_at(A1_diag,BdNodeidx,
                                normal[local_n2f_norm[:,0]][:,0])[isBdNode]
            A2_diag = bm.set_at(A2_diag,BdNodeidx,
                                normal[local_n2f_norm[:,0]][:,1])[isBdNode]
            A1 = spdiags(A1_diag ,0 , BDNN , BDNN,format='csr')
            A2 = spdiags(A2_diag ,0 , BDNN , BDNN,format='csr')

            VNN = (isVertex).sum()
            Vbd1_diag = normal[local_n2f_norm[isVertex,1]][:,0]
            Vbd2_diag = normal[local_n2f_norm[isVertex,1]][:,1]
            Vbd_constraint1 = bm.zeros((VNN,NN),dtype=bm.float64)
            Vbd_constraint2 = bm.zeros((VNN,NN),dtype=bm.float64)
            K = BdNodeidx[isVertex]
            Vbd_constraint1 = bm.set_at(Vbd_constraint1,
                                         (bm.arange(VNN),K),Vbd1_diag)
            Vbd_constraint2 = bm.set_at(Vbd_constraint2,
                                         (bm.arange(VNN),K),Vbd2_diag)

            A = bmat([[A1,A2],[Vbd_constraint1[:,isBdNode],
                               Vbd_constraint2[:,isBdNode]]],format='csr')

        elif self.TD == 3:    
            # 此处为边界局部信息,非全局
            isBdinnernode = local_n2f_norm[:,1] < 0
            isArrisnode = (local_n2f_norm[:,1] >= 0) & (local_n2f_norm[:,2] < 0)
            isVertex = local_n2f_norm[:,2] >= 0

            logic_Bdnode = logic_node[BdNodeidx]
            b_val0 = bm.sum(logic_Bdnode*normal[local_n2f_norm[:,0]],axis=1)
            b_val1 = bm.sum(logic_Bdnode[~isBdinnernode]*
                            normal[local_n2f_norm[~isBdinnernode,1]],axis=1)
            b_val2 = bm.sum(logic_Bdnode[isVertex]*
                            normal[local_n2f_norm[isVertex,2]],axis=1)
            b = bm.concatenate([b_val0,b_val1,b_val2])

            A1_diag = normal[local_n2f_norm[:,0]][:,0]
            A2_diag = normal[local_n2f_norm[:,0]][:,1]
            A3_diag = normal[local_n2f_norm[:,0]][:,2]

            A1 = spdiags(A1_diag ,0 , BDNN , BDNN,format='csr')
            A2 = spdiags(A2_diag ,0 , BDNN , BDNN,format='csr')
            A3 = spdiags(A3_diag ,0 , BDNN , BDNN,format='csr')
            BDBDNN = (~isBdinnernode).sum()
            K1 = bm.arange(BDNN,dtype=bm.int32)[~isBdinnernode]
            Bdbd1_diag = normal[local_n2f_norm[~isBdinnernode,1]][:,0]
            Bdbd2_diag = normal[local_n2f_norm[~isBdinnernode,1]][:,1]
            Bdbd3_diag = normal[local_n2f_norm[~isBdinnernode,1]][:,2]
            Bdbd_constraint1 = bm.zeros((BDBDNN,BDNN),dtype=bm.float64)
            Bdbd_constraint2 = bm.zeros((BDBDNN,BDNN),dtype=bm.float64)
            Bdbd_constraint3 = bm.zeros((BDBDNN,BDNN),dtype=bm.float64)
            Bdbd_constraint1 = bm.set_at(Bdbd_constraint1,
                                         (bm.arange(BDBDNN),K1),Bdbd1_diag)
            Bdbd_constraint2 = bm.set_at(Bdbd_constraint2,
                                         (bm.arange(BDBDNN),K1),Bdbd2_diag)
            Bdbd_constraint3 = bm.set_at(Bdbd_constraint3,
                                         (bm.arange(BDBDNN),K1),Bdbd3_diag)
            VNN = isVertex.sum()
            K2 = bm.arange(BDNN,dtype=bm.int32)[isVertex]
            Vbd1_diag = normal[local_n2f_norm[isVertex,2]][:,0]
            Vbd2_diag = normal[local_n2f_norm[isVertex,2]][:,1]
            Vbd3_diag = normal[local_n2f_norm[isVertex,2]][:,2]
            Vertex_constraint1 = bm.zeros((VNN,BDNN),dtype=bm.float64)
            Vertex_constraint2 = bm.zeros((VNN,BDNN),dtype=bm.float64)
            Vertex_constraint3 = bm.zeros((VNN,BDNN),dtype=bm.float64)
            Vertex_constraint1 = bm.set_at(Vertex_constraint1,
                                          (bm.arange(VNN),K2),Vbd1_diag)
            Vertex_constraint2 = bm.set_at(Vertex_constraint2,
                                          (bm.arange(VNN),K2),Vbd2_diag)
            Vertex_constraint3 = bm.set_at(Vertex_constraint3,
                                          (bm.arange(VNN),K2),Vbd3_diag)
            A = bmat([[A1,A2,A3],[Bdbd_constraint1,Bdbd_constraint2,Bdbd_constraint3],
                      [Vertex_constraint1,Vertex_constraint2,Vertex_constraint3]],format='csr')

        return A,b 
    
    def solve(self):
        """
        交替求解逻辑网格点
        logic_node : 新逻辑网格点
        vector_field : 逻辑网格点移动向量场
        """
        isBdNode = self.isBdNode
        logic_node = self.logic_node
        H11,H12,H21,H22 = self.get_stiff_matrix()
        A,b= self.A,self.b

        # 获得一个初始逻辑网格点的拷贝
        init_logic_node = logic_node.copy()
        process_logic_node = logic_node.copy()
        if self.redistribute:
            process_logic_node = self.redistribute_boundary()
        # 移动逻辑网格点
        f1 = -H12@process_logic_node[isBdNode,0]
        f2 = -H12@process_logic_node[isBdNode,1]
        
        move_innerlogic_node_x = spsolve(H11, f1)
        move_innerlogic_node_y = spsolve(H11, f2)
        process_logic_node = bm.set_at(process_logic_node , ~isBdNode, 
                                bm.stack([move_innerlogic_node_x,
                                          move_innerlogic_node_y],axis=1))
        
        f1 = -H21@move_innerlogic_node_x
        f2 = -H21@move_innerlogic_node_y
        b0 = bm.concatenate((f1,f2,b),axis=0)

        A1 = block_diag((H22, H22),format='csr')
        zero_matrix = csr_matrix((A.shape[0],A.shape[0]),dtype=bm.float64)
        A0 = bmat([[A1,A.T],[A,zero_matrix]],format='csr')

        move_bdlogic_node = spsolve(A0,b0)[:2*H22.shape[0]]
        process_logic_node = bm.set_at(process_logic_node , isBdNode,
                                bm.stack((move_bdlogic_node[:H22.shape[0]],
                                        move_bdlogic_node[H22.shape[0]:]),axis=1))

        vector_field = init_logic_node - process_logic_node  
        return process_logic_node,vector_field

    def get_physical_node(self,vector_field,logic_node_move):
        """
        计算物理网格点
        """
        node = self.node
        cell = self.cell
        cm = self.cm

        A = (node[cell[:,1:]] - node[cell[:,0,None]]).transpose(0,2,1)
        B = (logic_node_move[cell[:,1:]] - logic_node_move[cell[:,0,None]]).transpose(0,2,1)

        grad_x_incell = (A@bm.linalg.inv(B)) * cm[:,None,None]


        grad_x = bm.zeros((self.NN,2,2),dtype=bm.float64)
        bm.add_at(grad_x , self.i , grad_x_incell[self.j])
        grad_x /= self.star_measure[:,None,None]

        delta_x = (grad_x @ vector_field[:,:,None]  ).reshape(-1,2)

        local_n2f_norm = self.local_n2f_norm
        bdnode_normal = self.normal[local_n2f_norm[:,0]]
        bdtangent = bdnode_normal @ self.W
        BdNodeidx = self.BdNodeidx
        dot = bm.sum(bdtangent * delta_x[BdNodeidx],axis=1)
        delta_x = bm.set_at(delta_x,BdNodeidx,dot[:,None] * bdtangent)
        # 物理网格点移动距离
        C = (delta_x[cell[:,1:]] - delta_x[cell[:,0,None]]).transpose(0,2,1)
        a = C[:,0,0]*C[:,1,1] - C[:,0,1]*C[:,1,0]
        b = A[:,0,0]*C[:,1,1] - A[:,0,1]*C[:,1,0] + C[:,0,0]*A[:,1,1] - C[:,0,1]*A[:,1,0]
        c = A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]
        discriminant = b**2 - 4*a*c
        discriminant = bm.where(discriminant > 0, discriminant, 0)
        x1 = (-b + bm.sqrt(discriminant))/(2*a)
        x2 = (-b - bm.sqrt(discriminant))/(2*a)
        positive_x1 = bm.where(x1 > 0, x1, bm.inf)
        positive_x2 = bm.where(x2 > 0, x2, bm.inf)
        eta = bm.min([bm.min(positive_x1),bm.min(positive_x2),1])

        node = node + self.alpha * eta * delta_x
        return node
    
    def redistribute_boundary(self):
        """
        预处理边界节点
        """
        node = self.node
        logic_node = self.logic_node.copy()

        local_n2f_norm = self.local_n2f_norm
        vertex_idx = self.BdNodeidx[local_n2f_norm[:,-1] >= 0]
        BDNN = self.BDNN
        VNN = len(vertex_idx)
        K = bm.arange(BDNN,dtype=bm.int32)[local_n2f_norm[:,-1] >= 0]

        isBdedge = self.mesh.boundary_face_flag()
        node2edge = TM(self.node, self.cell).ds.node_to_edge()
        edge2cell = self.mesh.face_to_cell()
        for i in range(VNN):
            G = bm.arange(K[i]+1,K[i+1] if i < VNN-1 else BDNN)
            side_node_idx_bar = bm.concatenate(([vertex_idx[i]],
                                            self.BdNodeidx[G],
                                            [vertex_idx[(i+1)%VNN]]))
            side_node2edge = node2edge[self.BdNodeidx[G]][:,isBdedge]
            i,j = bm.nonzero(side_node2edge)
            _,k = bm.unique(j,return_index=True)
            j = j[bm.sort(k)]
            side_cell_idx = edge2cell[isBdedge][j][:,0]
            side_G = self.G[side_cell_idx]

            NN = side_node_idx_bar.shape[0]
            side_node = node[side_node_idx_bar]
            side_length = bm.linalg.norm(side_node[-1] - side_node[0])
            logic_side_node = logic_node[side_node_idx_bar]

            direction = logic_side_node[-1] - logic_side_node[0]
            angle = bm.arctan2(direction[1],direction[0])
            rotate = bm.array([[bm.cos(-angle),-bm.sin(-angle)],
                            [bm.sin(-angle),bm.cos(-angle)]])
            rate =bm.linalg.norm(direction)/side_length

            x = bm.linalg.norm(side_node - side_node[0],axis=1)
            cell = bm.stack([bm.arange(NN-1),bm.arange(1,NN)],axis=1)
            side_mesh = IntervalMesh(x , cell)
            space = LagrangeFESpace(side_mesh, p=1)
            qf = side_mesh.integrator(q=3)
            bcs, ws = qf.get_quadrature_points_and_weights()
            gphi = space.grad_basis(bcs)
            cell2dof = space.cell_to_dof()
            GDOF = space.number_of_global_dofs()
            H = bm.einsum('q , cqid , c ,cqjd, c -> cij ',ws, gphi ,side_G , gphi, side_mesh.entity_measure('cell'))
            I = bm.broadcast_to(cell2dof[:, :, None], shape=H.shape)
            J = bm.broadcast_to(cell2dof[:, None, :], shape=H.shape)
            H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
            lform = LinearForm(space)
            lform.add_integrator(ScalarSourceIntegrator(source=0,q=3))
            F = lform.assembly()
            bdIdx = bm.zeros(NN , dtype= bm.int32)
            bdIdx[[0,-1]] = 1
            D0 = spdiags(1-bdIdx ,0, NN, NN)
            D1 = spdiags(bdIdx , 0 , NN, NN)
            H = D0@H + D1
            F[[0,-1]] = x[[0,-1]]
            x = spsolve(H,F)  
            logic_side_node = logic_side_node[0] + rate * \
                                bm.stack([x,bm.zeros_like(x)],axis=1) @ rotate
            logic_node = bm.set_at(logic_node , side_node_idx_bar[1:-1] , logic_side_node[1:-1])

        return logic_node
    
    # 更新插值
    def interpolate(self,move_node):
        """
        @breif 将解插值到新网格上
        """
        delta_x = self.node - move_node
        mesh0 = TriangleMesh(move_node,self.cell)
        space = LagrangeFESpace(mesh0, p=1)
        qf = mesh0.integrator(3,'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        cm = mesh0.entity_measure('cell')
        phi = space.basis(bcs)
        H = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi , cm)   
        gphi = space.grad_basis(bcs)
        G = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi,  delta_x[self.cell], phi, cm)
        GDOF = space.number_of_global_dofs()
        I = bm.broadcast_to(space.cell_to_dof()[:, :, None], shape=G.shape)
        J = bm.broadcast_to(space.cell_to_dof()[:, None, :], shape=G.shape)
        H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
        G = csr_matrix((G.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
        def ODEs(t,y):
            f = spsolve(H,G@y)
            return f
        # 初值条件  
        uh0 = self.uh
        # 范围
        tau_span = [0,1]
        # 求解
        sol = solve_ivp(ODEs,tau_span,uh0,method='RK23')
        return sol.y[:,-1]
    
    def construct(self,new_mesh):
        """
        @breif 重构信息
        @param new_mesh 新的网格
        """
        self.mesh = new_mesh
        # node 更新之前完成插值
        self.uh = self.interpolate(new_mesh.entity('node'))
        self.node = new_mesh.entity('node')
        self.cm = new_mesh.entity_measure('cell')

        self.star_measure = self.get_star_measure()[0]

        self.G = self.get_control_function(self.beta,self.mol_times)

    def solve_elliptic_Equ(self,tol = None , maxit = 100):
        """
        @breif 求解椭圆方程
        @param pde PDEData
        @param tol 容许误差
        @param maxit 最大迭代次数
        """
         # 计算容许误差
        em = self.logic_mesh.entity_measure('edge')
        if tol is None:
            tol = bm.min(em)* 0.1
            print(f'容许误差为{tol}')

        init_logic_node = self.logic_node.copy()

        for i in range(maxit):
            logic_node,vector_field = self.solve()
            L_infty_error = bm.max(bm.linalg.norm(init_logic_node - logic_node,axis=1))
            node = self.get_physical_node(vector_field,logic_node)
            mesh0 = TriangleMesh(node,self.cell)
            print(f'第{i+1}次迭代的差值为{L_infty_error}')
            if L_infty_error < tol:
                self.construct(mesh0)
                print(f'迭代总次数:{i+1}次')
                return mesh0
            elif i == maxit - 1:
                print('超出最大迭代次数')
                break
            else:
                self.construct(mesh0)
                init_logic_node = logic_node.copy()