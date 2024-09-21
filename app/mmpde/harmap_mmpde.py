from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh as TM
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


class LogicMesh():
    def __init__(self , mesh: TriangleMesh):
        """
        mesh : 物理网格
        """
        self.mesh = mesh
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')
        self.edge = mesh.entity('edge')

        self.bdnodeindx = mesh.boundary_node_index()
        self.bdedgeindx = mesh.boundary_face_index()
        # 新网格下没有该方法
        self.node2edge = TM(self.node, self.cell).ds.node_to_edge()

        self.side_node_idx,\
        self.vertex_idx,\
        self.bdtangent = self.get_side_vertex_tangent()

    def __call__(self) :
        if self.is_convex():
            logic_mesh = TriangleMesh(self.node.copy(),self.cell) # 更新
        else:
            logic_node = self.get_logic_node()
            logic_cell = self.cell
            logic_mesh = TriangleMesh(logic_node,logic_cell)
        return logic_mesh
    
    def is_convex(self):
        """
        判断边界是否是凸的
        """
        vertex_idx = self.vertex_idx
        node = self.node
        intnode = node[vertex_idx]
        v0 = bm.roll(intnode,-1,axis=0) - intnode
        v1 = bm.roll(v0,-1,axis=0) 
        cross = bm.cross(v0,v1)
        return bm.all(cross > 0)

    def get_side_vertex_tangent(self):
        """
        side_node_idx : 每条边界上的节点全局索引的字典
        vertex_idx : 角点全局索引
        bdtangent : 每个边界节点的切向
        """
        bdnodeindx = self.bdnodeindx
        bdedgeindx = self.bdedgeindx   
        node = self.node
        edge = self.edge
        # 对边界边和点进行排序
        node2edge = self.node2edge
        bdnode2edge = node2edge[bdnodeindx][:,bdedgeindx]
        i,j = bm.nonzero(bdnode2edge)
        bdnode2edge = j.reshape(-1,2)
        glob_bdnode2edge = bm.zeros_like(node,dtype=bm.int32)
        glob_bdnode2edge = bm.set_at(glob_bdnode2edge,bdnodeindx,bdedgeindx[bdnode2edge])
        
        sort_glob_bdedge_idx_list = []
        sort_glob_bdnode_idx_list = []

        start_bdnode_idx = bdnodeindx[0]
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
                    remian_bdnode_idx = list(set(bdnodeindx)-set(sort_glob_bdnode_idx_list))
                    start_bdnode_idx = remian_bdnode_idx[0]
                    next_node_idx = start_bdnode_idx
                else:
                # 闭环跳出循环
                    break
            sort_glob_bdnode_idx_list.append(next_node_idx)
            current_node_idx = next_node_idx

        sort_glob_bdnode_idx = bm.array(sort_glob_bdnode_idx_list,dtype=bm.int32)
        # 获得切向
        bdtangent0 = node[edge[sort_glob_bdedge_idx_list]][:,1,:]\
                    -node[edge[sort_glob_bdedge_idx_list]][:,0,:]
        # 单位化
        norm = bm.linalg.norm(bdtangent0,axis = 1)
        bdtangent0 /= norm[:,None]
        # 原排列切向
        bdtangent = bm.zeros_like(node,dtype=bm.float64)
        bdtangent = bm.set_at(bdtangent,sort_glob_bdnode_idx,bdtangent0)[bdnodeindx]

        # 获得角点索引
        vertex_idx = bm.where(bm.sum(bm.abs(bdtangent0 - bm.roll(bdtangent0,1,axis=0)),axis=1)
                                > 1e-8)[0]
        side_node_idx = {}
        num_sides = len(vertex_idx)
        for i in range(num_sides):
            if i == num_sides - 1:
                # 处理最后一个索引，闭环情况
                side_node_idx[f'side{i}'] = bm.concatenate(
                    (sort_glob_bdnode_idx[vertex_idx[i]+1:],
                    sort_glob_bdnode_idx[:vertex_idx[(i+1)%num_sides]]))

            else:
                side_node_idx[f'side{i}'] = sort_glob_bdnode_idx[vertex_idx[i]+1:
                                                vertex_idx[(i+1)%num_sides]]
        
        vertex_idx = sort_glob_bdnode_idx[vertex_idx]
        return side_node_idx,vertex_idx,bdtangent
    
    def get_boundary_condition(self,p) -> TensorLike:
        """
        逻辑网格的边界条件
        """
        node = self.node
        side_node_idx = self.side_node_idx
        vertex_idx = self.vertex_idx
        physics_domain = node[vertex_idx]

        num_sides = physics_domain.shape[0]
        angles = bm.linspace(0,2*bm.pi,num_sides,endpoint=False)

        logic_domain = bm.stack([bm.cos(angles),bm.sin(angles)],axis=1)

        logic_bdnode = bm.zeros_like(node,dtype=bm.float64)
        logic_bdnode = bm.set_at(logic_bdnode,vertex_idx,logic_domain)
        
        Pside_length = bm.linalg.norm(bm.roll(physics_domain,-1,axis=0)
                                - physics_domain,axis=1)

        for i in range(num_sides):
            side_node = node[side_node_idx[f'side{i}']]
            side_part_length = bm.linalg.norm(side_node - physics_domain[i]
                                            ,axis=1)
            rate = (side_part_length/Pside_length[i]).reshape(-1,1)

            logic_bdnode = bm.set_at(logic_bdnode,side_node_idx[f'side{i}'],(1-rate)*logic_domain[i]\
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
        lform.add_integrator(ScalarSourceIntegrator(source=lambda x:0,q=p+1))
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
    

class Harmap_MMPDE():
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
        self.mesh = mesh
        self.uh = uh
        self.pde = pde
        self.beta = beta
        self.alpha = alpha
        self.mol_times = mol_times
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')

        self.cm = mesh.entity_measure('cell')
        # 新网格下没有该方法
        self.node2cell = TM(self.node, self.cell).ds.node_to_cell()
        self.isBdNode = mesh.boundary_node_flag()
        self.redistribute = redistribute

        self.W = bm.array([[0,1],[-1,0]],dtype=bm.int32)
        self.localEdge = bm.array([[1,2],[2,0],[0,1]],dtype=bm.int32)

        LM = LogicMesh(mesh)
        self.logic_mesh = LM()
        self.logic_node = self.logic_mesh.entity('node')
        self.side_node_idx = LM.side_node_idx
        self.vertex_idx = LM.vertex_idx
        self.bdtangent = LM.bdtangent
        self.isconvex = LM.is_convex()

        self.star_measure,self.i,self.j = self.get_star_measure()
        self.G = self.get_control_function(beta,mol_times)
        self.A , self.b = self.get_linear_constraint()

    def __call__(self):
        logic_node,vector_field = self.solve()
        node = self.get_physical_node(vector_field,logic_node)
        mesh = TriangleMesh(node,self.cell)
        return mesh
    

    def get_star_measure(self)->TensorLike:
        """
        计算每个节点的星的测度
        """
        star_measure = bm.zeros_like(self.node[:,0],dtype=bm.float64)
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
        side_node_idx = self.side_node_idx
        vertex_idx = self.vertex_idx
        isBdNode = self.isBdNode
        LNN = len(self.logic_node)
        
        logic_intnode = logic_node[vertex_idx]
        logic_bdtangent = bm.roll(logic_intnode,-1,axis=0) - logic_intnode
        logic_bdnormal = logic_bdtangent @ self.W

        b = bm.sum(logic_bdnormal * logic_intnode,axis=1)
        b_bdnode = bm.zeros(LNN , dtype=bm.float64)
        b_bdnode = bm.set_at(b_bdnode,vertex_idx,b)
        A1_diag = bm.zeros(LNN , dtype=bm.float64)
        A2_diag = bm.zeros(LNN , dtype=bm.float64)
        A1_diag = bm.set_at(A1_diag,vertex_idx,logic_bdnormal[:,0])
        A2_diag = bm.set_at(A2_diag,vertex_idx,logic_bdnormal[:,1])

        for i in range(len(side_node_idx)):
            A1_diag = bm.set_at(A1_diag,side_node_idx[f'side{i}'],logic_bdnormal[i,0])
            A2_diag = bm.set_at(A2_diag,side_node_idx[f'side{i}'],logic_bdnormal[i,1])
            b_bdnode = bm.set_at(b_bdnode,side_node_idx[f'side{i}'],b[i])

        A1 = spdiags(A1_diag ,0 , LNN , LNN,format='csr')[isBdNode][:,isBdNode]
        A2 = spdiags(A2_diag ,0 , LNN , LNN,format='csr')[isBdNode][:,isBdNode]
        A = hstack([A1,A2])
        b = b_bdnode[isBdNode]

        return A,b 
    
    def solve(self):
        """
        交替求解逻辑网格点
        logic_node : 新逻辑网格点
        vector_field : 逻辑网格点移动向量场
        """
        isBdNode = self.isBdNode
        logic_node = self.logic_node
        vertex_idx = self.vertex_idx
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

        process_logic_node = bm.set_at(process_logic_node , 
                                       vertex_idx,init_logic_node[vertex_idx])
        vector_field = init_logic_node - process_logic_node  
        return logic_node,vector_field

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

        i,j = bm.nonzero(self.node2cell)
        grad_x = bm.zeros((node.shape[0],2,2),dtype=bm.float64)
        bm.add_at(grad_x , i , grad_x_incell[j])
        grad_x /= self.star_measure[:,None,None]

        delta_x = (grad_x @ vector_field[:,:,None]  ).reshape(-1,2)

        bdtangent = self.bdtangent
        isBdNode = self.isBdNode
        dot = bm.sum(bdtangent * delta_x[isBdNode],axis=1)
        delta_x[isBdNode] =  dot[:,None] * bdtangent
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
        logic_node = self.logic_node
        side_node_idx = self.side_node_idx
        vertex_idx = self.vertex_idx
        isBdedge = self.mesh.boundary_edge_flag()
        node2edge = self.mesh.node_to_edge()
        edge2cell = self.mesh.edge_to_cell()
        # 每条边上做一维等分布,采用差分方式
        for i in range(len(side_node_idx)):
            side_node_idx_bar = bm.concatenate(([vertex_idx[i]],
                                            side_node_idx[f'side{i}'],
                                            [vertex_idx[(i+1)%len(side_node_idx)]]))
            side_node2edge = node2edge[side_node_idx[f'side{i}']][:,isBdedge]
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
            H = bm.einsum('q , qcid , c ,qcjd, c -> cij ',ws, gphi ,side_G , gphi, side_mesh.entity_measure('cell'))
            I = bm.broadcast_to(cell2dof[:, :, None], shape=H.shape)
            J = bm.broadcast_to(cell2dof[:, None, :], shape=H.shape)
            H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
            lform = LinearForm(space)
            lform.add_domain_integrator(ScalarSourceIntegrator(f=lambda x:0,q=3))
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
        H = bm.einsum('q , qci ,qcj, c -> cij ',ws, phi ,phi , cm)   
        gphi = space.grad_basis(bcs)
        G = bm.einsum('q , qcid , cid , qcj ,c -> cij' , ws , gphi,  delta_x[self.cell], phi, cm)
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
        self.uh = self.interpolate(self.node)
        self.node = new_mesh.entity('node')
        self.cm = new_mesh.entity_measure('cell')

        self.star_measure = self.get_star_measure(self.mesh)[0]

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
            print(f'第{i+1}次迭代的差值为{L_infty_error}')
            if L_infty_error < tol:
                node = self.get_physical_node(vector_field,logic_node)
                mesh0 = TriangleMesh(node,self.cell)
                print(f'迭代总次数:{i+1}次')
                return mesh0
            elif i == maxit - 1:
                print('超出最大迭代次数')
                break
            else:
                node = self.get_physical_node(vector_field,logic_node)
                mesh0 = TriangleMesh(node,self.cell)
                self.construct(mesh0)
                init_logic_node = logic_node.copy()