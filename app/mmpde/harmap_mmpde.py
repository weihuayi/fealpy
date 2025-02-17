from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import IntervalMesh
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.old.mesh import TriangleMesh as TM
from fealpy.old.mesh import TetrahedronMesh as THM
from fealpy.functionspace import LagrangeFESpace,ParametricLagrangeFESpace
from fealpy.fem import (BilinearForm 
                        ,ScalarDiffusionIntegrator
                        ,LinearForm
                        ,ScalarSourceIntegrator
                        ,DirichletBC)
from fealpy.solver import spsolve
from scipy.sparse.linalg import spsolve as spsolve1
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix,spdiags,block_diag,bmat
from sympy import *
import matplotlib.pyplot as plt
from typing import Any ,Union,Optional
import pyamg


class LogicMesh():
    def __init__(self , mesh: Union[TriangleMesh,TetrahedronMesh,LagrangeTriangleMesh],
                        Vertex_idx : TensorLike,
                        Bdinnernode_idx : TensorLike,
                        Arrisnode_idx : Optional[TensorLike] = None,
                        sort_BdNode_idx : Optional[TensorLike] = None) -> None:
        """
        @param mesh:  物理网格
        @param Vertex_idx : 角点全局编号
        @param Bdinnernode_idx : 面内点全局编号
        @param Arrisnode_idx : 棱内点全局编号
        @param sort_BdNode_idx : 排序后的边界点全局编号
        """
        self.mesh = mesh
        self.TD = mesh.top_dimension()
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')
        self.isinstance_mesh_type(mesh)

        self.BdNodeidx = mesh.boundary_node_index()
        self.BdFaceidx = mesh.boundary_face_index()
        self.Vertex_idx = Vertex_idx
        self.Bdinnernode_idx = Bdinnernode_idx
        self.sort_BdNode_idx = sort_BdNode_idx
        self.roll_SortBdNode()
        self.isconvex = self.is_convex()
        if self.isconvex == False:
            if sort_BdNode_idx is None:
                raise ValueError('The boundary is not convex, you must give the sort_BdNode')
            
        self.logic_mesh = self.get_logic_mesh()
        self.logic_node = self.logic_mesh.entity('node')
        self.isconvex = self.is_convex()
        # 新网格下没有该方法
        if self.TD == 2:
            if self.mesh_type == "LagrangeTriangleMesh":
                self.node2edge = TM(bm.to_numpy(self.linermesh.node), 
                                    bm.to_numpy(self.linermesh.cell)).ds.node_to_edge()
            else:
                self.node2edge = TM(bm.to_numpy(self.node), 
                                bm.to_numpy(self.cell)).ds.node_to_edge()
            self.Bi_Lnode_normal = self.get_normal_information(self.logic_mesh)
            self.Bi_Pnode_normal = self.get_normal_information(self.mesh)

        if self.TD == 3:
            if Arrisnode_idx is None:
                raise ValueError('TD = 3, you must give the Arrisnode_idx')
            self.Arrisnode_idx = Arrisnode_idx
            self.Bi_Lnode_normal, self.Ar_Lnode_normal = self.get_normal_information(self.logic_mesh)
        

    def isinstance_mesh_type(self,mesh):
        if isinstance(mesh, TriangleMesh):
            self.mesh_type = "TriangleMesh"
            self.mesh_class = TriangleMesh
            self.p = 1
        elif isinstance(mesh, TetrahedronMesh):
            self.mesh_type = "TetrahedronMesh"
            self.mesh_class = TetrahedronMesh
            self.p = 1
        elif isinstance(mesh, LagrangeTriangleMesh):
            self.mesh_type = "LagrangeTriangleMesh"
            self.linermesh = mesh.linearmesh
            self.mesh_class = LagrangeTriangleMesh
            self.p = mesh.p
        else:
            raise TypeError("Unsupported mesh type")
        
    def get_logic_mesh(self) :
        if not self.is_convex():
            if self.TD == 3:
                raise ValueError('非凸多面体无法构建逻辑网格')
            logic_node = self.get_logic_node()
            if self.mesh_type == "LagrangeTriangleMesh":
                logic_cell = self.linermesh.cell
                linear_logic_mesh = TriangleMesh(logic_node,logic_cell)
                logic_mesh = self.mesh_class.from_triangle_mesh(linear_logic_mesh, self.p)
            else:
                logic_cell = self.cell
                logic_mesh = self.mesh_class(logic_node,logic_cell)  
        else:
            if self.mesh_type == "LagrangeTriangleMesh":
                logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh, self.p)
            else:
                logic_mesh = self.mesh_class(bm.copy(self.node),self.cell)
        return logic_mesh
        
    def is_convex(self):
        """
        @brief is_convex : 判断边界是否是凸的
        """
        from scipy.spatial import ConvexHull
        intnode = self.node[self.Vertex_idx]
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
                        face.flatten(),
                        bm.repeat(bm.arange(NF), NVF)
                    )
                ), shape=(NN, NF))
        return node2face
    
    def get_normal_information(self,mesh:Union[TriangleMesh,
                                               TetrahedronMesh,
                                               LagrangeTriangleMesh]):
        """
        @brief get_normal_information: 获取边界点法向量
        """
        Bdinnernode_idx = self.Bdinnernode_idx
        BdFaceidx = self.BdFaceidx
        if self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            node2face = bm.tensor(self.node_to_face().todense())
            Ar_node2face = node2face[Arrisnode_idx][:,BdFaceidx]
            i0 , j0 = bm.nonzero(Ar_node2face)
            bdfun0 = mesh.face_unit_normal(index=BdFaceidx[j0])
            normal0,inverse0 = bm.unique(bdfun0,return_inverse=True ,axis = 0)
            _,index0,counts0 = bm.unique(i0,return_index=True,return_counts=True)   
            maxcount = bm.max(counts0)
            mincount = bm.min(counts0)
            Ar_node2normal_idx = -bm.ones((len(Arrisnode_idx),maxcount),dtype=bm.int32)
            Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,
                                            (slice(None),slice(mincount)),
                                            inverse0[index0[:,None]+bm.arange(mincount)])
            for i in range(maxcount-mincount):
                isaimnode = counts0 > mincount+i
                Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,(isaimnode,mincount+i) , 
                                                inverse0[index0[isaimnode]+mincount+i])
            for i in range(Ar_node2normal_idx.shape[0]):
                x = Ar_node2normal_idx[i]
                unique_vals = bm.unique(x[x >= 0])
                Ar_node2normal_idx[i, :len(unique_vals)] = unique_vals
            Ar_node2normal = normal0[Ar_node2normal_idx[:,:2]]
        elif self.TD == 2:
            node2face = bm.tensor(self.node2edge.todense())
            
        if self.mesh_type == "LagrangeTriangleMesh":
            LBdFace = mesh.face[BdFaceidx]
            LNN = mesh.number_of_nodes()
            LBd_node2face = bm.zeros((LNN , 2),  dtype=bm.int64)
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,:2],0) , BdFaceidx[:,None])
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,1:],1) , BdFaceidx[:,None])
            LBdi_node2face = LBd_node2face[Bdinnernode_idx]
            linear_mesh = mesh.linearmesh
            bdfun  = linear_mesh.face_unit_normal(index=LBdi_node2face[:,0])
            return bdfun
        Bi_node2face = node2face[Bdinnernode_idx][:,BdFaceidx]
        i1 , j1 = bm.nonzero(Bi_node2face)
        bdfun1 = mesh.face_unit_normal(index=BdFaceidx[j1])
        _,index1 = bm.unique(i1,return_index=True)
        Bi_node_normal = bdfun1[index1]
        if self.TD == 2:
            return Bi_node_normal
        else:
            return Bi_node_normal, Ar_node2normal
    
    def roll_SortBdNode(self):
        """
        对齐边界点与角点
        """
        sBdnodeidx = self.sort_BdNode_idx
        Vertexidx = self.Vertex_idx
        if sBdnodeidx is not None and sBdnodeidx[0] != Vertexidx[0]:
            K = bm.where(sBdnodeidx == Vertexidx[0])[0][0]
            self.sort_BdNode_idx = bm.roll(sBdnodeidx,-K)
        
    def get_boundary_condition(self,p) -> TensorLike:
        """
        逻辑网格的边界条件
        """
        if self.mesh_type == "LagrangeTriangleMesh":
            node = self.linermesh.node
            map = bm.where(self.sort_BdNode_idx < len(node))[0]
            sBdNodeidx = self.sort_BdNode_idx[map]
        else:
            node = self.node
            sBdNodeidx = self.sort_BdNode_idx 
        Vertexidx = self.Vertex_idx

        physics_domain = node[Vertexidx]
        num_sides = physics_domain.shape[0]
        angles = bm.linspace(0,2*bm.pi,num_sides,endpoint=False)
        logic_domain = bm.stack([bm.cos(angles),bm.sin(angles)],axis=1)
        logic_bdnode = bm.zeros_like(node,dtype=bm.float64)
        
        Pside_vector = bm.roll(physics_domain,-1,axis=0) - physics_domain
        Lside_vector = bm.roll(logic_domain,-1,axis=0) - logic_domain
        Pside_length = bm.linalg.norm(Pside_vector,axis=1)
        Lside_length = bm.linalg.norm(Lside_vector,axis=1)
        rate = Lside_length / Pside_length
        theta = bm.arctan2(Lside_vector[:,1],Lside_vector[:,0]) -\
                bm.arctan2(Pside_vector[:,1],Pside_vector[:,0]) 
        A = rate[:,None,None] * (bm.array([[bm.cos(theta),bm.sin(theta)],
                                           [-bm.sin(theta),bm.cos(theta)]],dtype=bm.float64)).T

        K = bm.where(sBdNodeidx[:,None] == Vertexidx)[0]
        K = bm.concatenate([K,[len(sBdNodeidx)]])
        A_repeat = bm.repeat(A,K[1:]-K[:-1],axis=0)
        PVertex_repeat = bm.repeat(physics_domain,K[1:]-K[:-1],axis=0)
        LVertex_repeat = bm.repeat(logic_domain,K[1:]-K[:-1],axis=0)
        Aim_vector = (A_repeat@((node[sBdNodeidx]-PVertex_repeat)[:,:,None])).reshape(-1,2)
        logic_bdnode = bm.set_at(logic_bdnode,sBdNodeidx,Aim_vector+LVertex_repeat)
        map = bm.where((node[:,None] == p).all(axis=2))[0]
        return logic_bdnode[map]
    
    def get_logic_node(self) -> TensorLike:
        """
        logic_node : 逻辑网格的节点坐标
        """
        if self.mesh_type == "LagrangeTriangleMesh":
            mesh = self.linermesh
        else:
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
        uh0 = space.function()
        uh1 = space.function()
        A1, F1 = bc0.apply(A, F, uh0)
        A2, F2 = bc1.apply(A, F, uh1)
        uh0 = bm.set_at(uh0 , slice(None), spsolve(A1, F1 , solver="scipy"))
        uh1 = bm.set_at(uh1 , slice(None), spsolve(A2, F2 , solver="scipy"))
        logic_node = bm.stack([uh0,uh1],axis=1)
        return logic_node

class Harmap_MMPDE(LogicMesh):
    def __init__(self, 
                 mesh:Union[TriangleMesh,TetrahedronMesh,LagrangeTriangleMesh] , 
                 uh:TensorLike,
                 beta :float ,
                 Vertex_idx : TensorLike,
                 Bdinnernode_idx : TensorLike,
                 Arrisnode_idx : Optional[TensorLike] = None,
                 sort_BdNode_idx : Optional[TensorLike] = None,
                 alpha = 0.5, 
                 mol_times = 3 , 
                 redistribute = False) -> None:
        """
        @param mesh: 初始物理网格
        @param uh: 物理网格上的解
        @param pde: 微分方程基本信息
        @param beta: 控制函数的参数
        @param alpha: 移动步长控制参数
        @param mol_times: 磨光次数
        @param redistribute: 是否预处理边界节点
        """
        super().__init__(mesh = mesh,
                         Vertex_idx = Vertex_idx,
                         Bdinnernode_idx = Bdinnernode_idx,
                         Arrisnode_idx = Arrisnode_idx,
                         sort_BdNode_idx = sort_BdNode_idx)
        self.uh = uh
        self.beta = beta
        self.alpha = alpha
        self.mol_times = mol_times
        self.BDNN = len(self.BdNodeidx)
        self.cm = mesh.entity_measure('cell')
        if self.mesh_type == "LagrangeTriangleMesh":
            self.NN = mesh.number_of_nodes()
            self.linerNN = self.linermesh.number_of_nodes()
            self.space = ParametricLagrangeFESpace(self.mesh, p=self.p)
        else:
            self.NN = mesh.number_of_nodes()
            self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.isBdNode = mesh.boundary_node_flag()
        self.redistribute = redistribute
        self.multi_index = bm.multi_index_matrix(self.p,self.TD)/self.p
        self.star_measure = self.get_star_measure()
        # self.indices,self.indptr = self.grad_simple()
        self.G,self.M = self.get_control_function()
        self.A , self.b = self.get_linear_constraint()
        if redistribute and sort_BdNode_idx is None:
            raise ValueError('redistributing boundary , you must give the sort_BdNode')
        self.tol = self.caculate_tol()

    def get_star_measure(self)->TensorLike:
        """
        计算每个节点的星的测度
        """
        NN = self.NN
        star_measure = bm.zeros(NN,dtype=bm.float64)
        bm.add_at(star_measure , self.cell , self.cm[:,None])
        return star_measure
    
    def get_control_function(self):
        """
        @brief 计算控制函数
        """
        cell = self.cell
        cm = self.cm
        multi_index = self.multi_index
        space = self.space
        if self.mesh_type == "LagrangeTriangleMesh":
            gphi = space.grad_basis(multi_index)
            guh_incell = bm.einsum('cqid , ci -> cqd ',gphi , self.uh[cell]) # (NC,ldof,GD)
            guh_innode = bm.zeros((self.NN,2),dtype=bm.float64)
            bm.add_at(guh_innode , cell , cm[:,None,None]*guh_incell)
            guh_innode /= self.star_measure[:,None]
            max_norm_guh = bm.max(bm.linalg.norm(guh_innode,axis=1))
            M = bm.sqrt(1 + self.beta * bm.sum(guh_innode**2,axis=1)/max_norm_guh**2) # (NN,)
            for k in range(self.mol_times):
                M_incell = bm.mean(M[cell],axis=1)
                M = bm.zeros(self.NN,dtype=bm.float64)
                bm.add_at(M , cell, (cm *M_incell)[: , None])
                M /= self.star_measure
            qf = self.mesh.quadrature_formula(self.p+1)
            bcs = qf.get_quadrature_points_and_weights()[0]
            phi = self.mesh.shape_function(bc = bcs,p = self.p)
            M_incell = bm.einsum('cqi , ci  -> cq ', phi , M[cell])

        else:
            gphi = space.grad_basis(multi_index)[:,0,...]
            cell2dof = space.cell_to_dof()
            guh_incell = bm.einsum('cid , ci -> cd ',gphi , self.uh[cell2dof]) # (NC,ldof,GD)
            max_norm_guh = bm.max(bm.linalg.norm(guh_incell,axis=1))
            if max_norm_guh == 0:
                max_norm_guh = 1
            M_incell = bm.sqrt(1 +self.beta *bm.sum(guh_incell**2,axis=1)/max_norm_guh)
            if self.mol_times > 0:
                for k in range(self.mol_times):
                    M = bm.zeros(self.NN,dtype=bm.float64)
                    bm.add_at(M , cell, (cm *M_incell)[: , None])
                    M /= self.star_measure
                    M_incell = bm.mean(M[cell],axis=1)
        return 1/M_incell,1/M
    
    def get_stiff_matrix(self,mesh:Union[TriangleMesh,TetrahedronMesh,LagrangeTriangleMesh],G:TensorLike):
        """
        @brief 组装刚度矩阵
        @param mesh: 物理网格
        @param G: 控制函数
        """
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(self.p+1)
        bcs, ws = qf.get_quadrature_points_and_weights()
        space = self.space
        gphi = space.grad_basis(bcs)
        # cell2dof = space.cell_to_dof()
        # GDOF = space.number_of_global_dofs()
        cell2dof = mesh.entity('cell')
        GDOF = self.node.shape[0]
        if self.mesh_type == "LagrangeTriangleMesh":
            # rm = mesh.reference_cell_measure()
            # J = mesh.jacobi_matrix(bcs)
            # d = rm * bm.linalg.det(J)
            H = bm.einsum('q , cqid , cq ,cqjd, c -> cij ',ws, gphi ,G , gphi,cm)
        else:
            H = bm.einsum('q , cqid , c ,cqjd, c -> cij ',ws, gphi ,G , gphi, cm)
        I = bm.broadcast_to(cell2dof[:, :, None], shape=H.shape)
        J = bm.broadcast_to(cell2dof[:, None, :], shape=H.shape)
        H = csr_matrix((H.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
        return H
    
    def get_linear_constraint(self):
        """
        @brief 组装线性约束
        """
        logic_node = self.logic_node
        BdNodeidx = self.BdNodeidx
        Vertex_idx = self.Vertex_idx
        Bdinnernode_idx = self.Bdinnernode_idx
        Binnorm = self.Bi_Lnode_normal
        logic_Bdinnode = logic_node[Bdinnernode_idx]
        logic_Vertex = logic_node[Vertex_idx]
        NN = self.NN
        BDNN = self.BDNN
        VNN = len(Vertex_idx)

        b = bm.zeros(NN, dtype=bm.float64)
        b_val0 = bm.sum(logic_Bdinnode * Binnorm, axis=1)
        b[Bdinnernode_idx] = b_val0

        A_diag = bm.zeros((self.TD , NN)  , dtype=bm.float64)
        A_diag[:,Bdinnernode_idx] = Binnorm.T
        A_diag[0,Vertex_idx] = 1
        if self.TD == 2:
            b[Vertex_idx] = logic_Vertex[:, 0]
            b = bm.concatenate([b[BdNodeidx],logic_Vertex[:,1]])
            A_diag = A_diag[:,BdNodeidx]

            A = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                       spdiags(A_diag[1], 0, BDNN, BDNN, format='csr')],
                      [csr_matrix((VNN, BDNN), dtype=bm.float64),
                       csr_matrix((bm.ones(VNN, dtype=bm.float64),
                                  (bm.arange(VNN), Vertex_idx)), 
                        shape=(VNN, NN))[:, BdNodeidx]]], format='csr')

        elif self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            Arnnorm = self.Ar_Lnode_normal
            logic_Arnode = logic_node[Arrisnode_idx]
            ArNN = len(Arrisnode_idx)
            b_val1 = bm.sum(logic_Arnode*Arnnorm[:,0,:],axis=1)
            b_val2 = bm.sum(logic_Arnode*Arnnorm[:,1,:],axis=1)
            b = bm.set_at(b , Arrisnode_idx , b_val1)
            b = bm.set_at(b , Vertex_idx , logic_Vertex[:,0])[BdNodeidx]
            b = bm.concatenate([b,b_val2,logic_Vertex[:,1],logic_Vertex[:,2]])       

            A_diag[:,Arrisnode_idx] = Arnnorm[:,0,:].T
            A_diag = A_diag[:,BdNodeidx]
            
            index1 = NN * bm.arange(self.TD) + Arrisnode_idx[:,None]
            index2 = NN * bm.arange(self.TD) + BdNodeidx[:,None]
            rol_Ar = bm.repeat(bm.arange(ArNN)[None,:],3,axis=0).flat
            cow_Ar = index1.T.flat
            data_Ar = Arnnorm[:,1,:].T.flat
            Ar_constraint = csr_matrix((data_Ar,(rol_Ar, cow_Ar)),shape=(ArNN,3*NN))
            Vertex_constraint1 = csr_matrix((bm.ones(VNN,dtype=bm.float64),
                                (bm.arange(VNN),Vertex_idx + NN)),shape=(VNN,3*NN))
            Vertex_constraint2 = csr_matrix((bm.ones(VNN,dtype=bm.float64),
                                (bm.arange(VNN),Vertex_idx + 2 * NN)),shape=(VNN,3*NN))

            A_part = bmat([[spdiags(A_diag[0], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[1], 0, BDNN, BDNN, format='csr'),
                            spdiags(A_diag[2], 0, BDNN, BDNN, format='csr')]], format='csr')
            A = bmat([[A_part],
                      [Ar_constraint[:, index2.T.flat]],
                      [Vertex_constraint1[:, index2.T.flat]],
                      [Vertex_constraint2[:, index2.T.flat]]], format='csr')
        return A,b

    def solve_move_LogicNode(self):
        """
        @brief 交替求解逻辑网格点
        @param process_logic_node: 新逻辑网格点
        @param move_vector_field: 逻辑网格点移动向量场
        """
        isBdNode = self.isBdNode
        TD = self.TD
        BDNN = self.BDNN
        INN = self.NN - BDNN
        H = self.get_stiff_matrix(self.mesh,self.G)
        H11 = H[~isBdNode][:, ~isBdNode]
        H12 = H[~isBdNode][:, isBdNode]
        H21 = H[isBdNode][:, ~isBdNode]
        H22 = H[isBdNode][:, isBdNode]

        A,b= self.A,self.b
        # 获得一个初始逻辑网格点的拷贝
        init_logic_node = self.logic_node.copy()
        process_logic_node = self.logic_node.copy()
        if self.redistribute:
            process_logic_node = self.redistribute_boundary()
        # 移动逻辑网格点
        F = -H12 @ process_logic_node[isBdNode, :]
        move_innerlogic_node = bm.zeros((INN, TD), dtype=bm.float64)
        ml = pyamg.ruge_stuben_solver(H11)
        for i in range(TD):
            move_innerlogic_node[:, i] = ml.solve(F[:, i] , tol=1e-8)
        process_logic_node = bm.set_at(process_logic_node, ~isBdNode, move_innerlogic_node)

        F = (-H21 @ move_innerlogic_node).T.flatten()
        b0 = bm.concatenate((F,b),axis=0)

        A1 = block_diag([H22]*TD,format='csr')
        zero_matrix = csr_matrix((A.shape[0],A.shape[0]),dtype=bm.float64)
        A0 = bmat([[A1,A.T],[A,zero_matrix]],format='csr')

        move_bdlogic_node = spsolve1(A0,b0)[:TD*BDNN]
        move_bdlogic_node = move_bdlogic_node.reshape((TD, BDNN)).T
        process_logic_node = bm.set_at(process_logic_node , isBdNode, move_bdlogic_node)
        move_vector_field = init_logic_node - process_logic_node

        return process_logic_node,move_vector_field

    def get_physical_node(self,move_vertor_field,logic_node_move):
        """
        @brief 计算物理网格点
        @param move_vertor_field: 逻辑网格点移动向量场
        @param logic_node_move: 移动后的逻辑网格点
        """
        node = self.node
        cell = self.cell
        cm = self.cm
        TD = self.TD
        p = self.p
        space = self.space
        multi_index = self.multi_index 
        gphi = space.grad_basis(multi_index)
        grad_X_incell = bm.einsum('cin, cqim -> cqnm',logic_node_move[cell], gphi)
        grad_x = bm.zeros((self.NN,2,2),dtype=bm.float64)
        grad_x_incell = bm.linalg.inv(grad_X_incell)*cm[:,None,None,None]
        bm.add_at(grad_x , cell , grad_x_incell)
        grad_x /= self.star_measure[:,None,None]
        delta_x = (grad_x @ move_vertor_field[:,:,None]).reshape(-1,2)

        if TD == 3:
            self.Bi_Pnode_normal = self.Bi_Lnode_normal
            Ar_Pnode_normal = self.Ar_Lnode_normal
            Arrisnode_idx = self.Arrisnode_idx
            dot1 = bm.sum(Ar_Pnode_normal * delta_x[Arrisnode_idx,None],axis=-1)
            delta_x[Arrisnode_idx] -= (dot1[:,0,None] * Ar_Pnode_normal[:,0,:] + 
                                       dot1[:,1,None] * Ar_Pnode_normal[:,1,:])
        
        Bdinnernode_idx = self.Bdinnernode_idx
        dot = bm.sum(self.Bi_Pnode_normal * delta_x[Bdinnernode_idx],axis=1)
        delta_x[Bdinnernode_idx] -= dot[:,None] * self.Bi_Pnode_normal
        A = (node[cell[:,1:]] - node[cell[:,0,None]]).transpose(0,2,1)
        C = (delta_x[cell[:,1:]] - delta_x[cell[:,0,None]]).transpose(0,2,1)
        # 物理网格点移动距离
        if TD == 2:
            if self.mesh_type == "LagrangeTriangleMesh":
                qf = self.mesh.quadrature_formula(p)
                bc,ws = qf.get_quadrature_points_and_weights()
                J = self.mesh.jacobi_matrix(bc=bc)
                gphi = self.mesh.grad_shape_function(bc=bc, variables='u')
                dJ = bm.einsum('cin, cqim -> cqnm',delta_x[cell],gphi)
                a = bm.linalg.det(dJ)@ws
                c = bm.linalg.det(J)@ws
                b = (J[...,0,0]*dJ[...,1,1] + J[...,1,1]*dJ[...,0,0] \
                   - J[...,0,1]*dJ[...,1,0] - J[...,1,0]*dJ[...,0,1])@ws             
            else:
                a = bm.linalg.det(C)
                c = bm.linalg.det(A)
                b = A[:,0,0]*C[:,1,1] - A[:,0,1]*C[:,1,0] + C[:,0,0]*A[:,1,1] - C[:,0,1]*A[:,1,0]
            discriminant = b**2 - 4*a*c
            right_idx = bm.where(discriminant >= 0)[0]
            x = bm.concatenate([(-b[right_idx] + bm.sqrt(discriminant[right_idx]))/(2*a[right_idx]),
                                (-b[right_idx] - bm.sqrt(discriminant[right_idx]))/(2*a[right_idx])])
        else:
            # 三维情况，求解三次方程
            a0,a1,a2 = C[:,1,1]*C[:,2,2] - C[:,1,2]*C[:,2,1],\
                       C[:,1,2]*C[:,2,0] - C[:,1,0]*C[:,2,2],\
                       C[:,1,0]*C[:,2,1] - C[:,1,1]*C[:,2,0]
            b0,b1,b2 = A[:,1,1]*C[:,2,2] - A[:,1,2]*C[:,2,1] + C[:,1,1]*A[:,2,2] - C[:,1,2]*A[:,2,1],\
                       A[:,1,0]*C[:,2,2] - A[:,1,2]*C[:,2,0] + C[:,1,0]*A[:,2,2] - C[:,1,2]*A[:,2,0],\
                       A[:,1,0]*C[:,2,1] - A[:,1,1]*C[:,2,0] + C[:,1,0]*A[:,2,1] - C[:,1,1]*A[:,2,0]
            c0,c1,c2 = A[:,1,1]*A[:,2,2] - A[:,1,2]*A[:,2,1],\
                       A[:,1,0]*A[:,2,2] - A[:,1,2]*A[:,2,0],\
                       A[:,1,0]*A[:,2,1] - A[:,1,1]*A[:,2,0]
            a = C[:,0,0]*a0 - C[:,0,1]*a1 + C[:,0,2]*a2
            b = A[:,0,0]*a0 - A[:,0,1]*a1 + A[:,0,2]*a2 + C[:,0,0]*b0 - C[:,0,1]*b1 + C[:,0,2]*b2
            c = A[:,0,0]*b0 - A[:,0,1]*b1 + A[:,0,2]*b2 + C[:,0,0]*c0 - C[:,0,1]*c1 + C[:,0,2]*c2
            d = A[:,0,0]*c0 - A[:,0,1]*c1 + A[:,0,2]*c2
            # 使用卡尔达诺公式求解三次方程的实根
            p = (3 * a * c - b**2) / (3 * a**2)
            q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
            discriminant = (q / 2)**2 + (p / 3)**3
            u = bm.where(discriminant >= 0, (-q / 2 + bm.sqrt(discriminant))**(1 / 3),
                                            (-q / 2 + bm.sqrt(-discriminant) * 1j)**(1 / 3))
            v = bm.where(discriminant >= 0, (-q / 2 - bm.sqrt(discriminant))**(1 / 3), 
                                            (-q / 2 - bm.sqrt(-discriminant) * 1j)**(1 / 3))
            move_dis = b / (3 * a)
            x1 = u + v - move_dis
            x2 = -(u + v) / 2 + (u - v) * bm.sqrt(3) * 1j / 2 - move_dis
            x3 = -(u + v) / 2 - (u - v) * bm.sqrt(3) * 1j / 2 - move_dis
            x = bm.concatenate([x1, x2, x3])
        positive_x = bm.where(x>0, x, 1)
        eta = bm.min(positive_x)
        node = node +  self.alpha*eta* delta_x
        return node
            
    
    def redistribute_boundary(self):
        """
        @brief 预处理边界节点
        """
        node = self.node
        logic_node = self.logic_node.copy()
        Vertex_idx = self.Vertex_idx
        sort_Bdnode_idx = self.sort_BdNode_idx
        K = bm.where(sort_Bdnode_idx[:,None] == Vertex_idx)[0]
        isBdedge = self.mesh.boundary_face_flag()
        node2edge = TM(self.node, self.cell).ds.node_to_edge()
        edge2cell = self.mesh.face_to_cell()
        G_cell = self.get_control_function(self.beta,mol_times=4)[0]
        
        VNN = len(Vertex_idx)
        for n in range(VNN):
            side_node_idx = sort_Bdnode_idx[K[n]:K[n+1]+1] \
                            if n < VNN - 1 else sort_Bdnode_idx[K[n]:]
            side_node2edge = node2edge[side_node_idx[1:-1]][:,isBdedge]
            i,j = bm.nonzero(side_node2edge)
            _,k = bm.unique(j,return_index=True)
            j = j[bm.sort(k)]
            side_cell_idx = edge2cell[isBdedge][j][:,0]
            side_G = G_cell[side_cell_idx]

            SNN = side_node_idx.shape[0]
            side_node = node[side_node_idx]
            side_length = bm.linalg.norm(side_node[-1] - side_node[0])
            logic_side_node = logic_node[side_node_idx]

            direction = logic_side_node[-1] - logic_side_node[0]
            angle = bm.arctan2(direction[1],direction[0])
            rotate = bm.array([[bm.cos(-angle),-bm.sin(-angle)],
                            [bm.sin(-angle),bm.cos(-angle)]])
            rate =bm.linalg.norm(direction)/side_length

            x = bm.linalg.norm(side_node - side_node[0],axis=1)
            cell = bm.stack([bm.arange(SNN-1),bm.arange(1,SNN)],axis=1)
            side_mesh = IntervalMesh(x , cell)
            H = self.get_stiff_matrix(side_mesh,side_G)
            F = bm.zeros(SNN , dtype= bm.float64)
            F = bm.set_at(F , [0,-1] , [x[0],x[-1]])
            bdIdx = bm.zeros(SNN , dtype= bm.float64)
            bdIdx = bm.set_at(bdIdx , [0,-1] , 1)
            D0 = spdiags(1-bdIdx ,0, SNN, SNN)
            D1 = spdiags(bdIdx , 0 , SNN, SNN)
            H = D0@H + D1
            x = spsolve1(H,F)
            logic_side_node = logic_side_node[0] + rate * \
                                bm.stack([x,bm.zeros_like(x)],axis=1) @ rotate
            logic_node = bm.set_at(logic_node , side_node_idx[1:-1] , logic_side_node[1:-1])
        return logic_node

    def interpolate(self,move_node):
        """
        @brief 将解插值到新网格上
        @param move_node: 移动后的物理节点
        """
        delta_x = self.node - move_node
        cell = self.cell
        mesh = self.mesh
        space = self.space
        qf = mesh.quadrature_formula(self.p+1,'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        if self.mesh_type == "LagrangeTriangleMesh":
            cell2dof = bm.concat([cell[:,0:3],cell[:,[1,3,4]],
                                  cell[:,[1,4,2]],cell[:,[2,4,5]]],axis = 0)
            # cell2dof = cell
            GDOF = self.NN
            mesh0 = TriangleMesh(self.node,cell2dof)
            space1 = LagrangeFESpace(mesh0, p=1)
            phi = space1.basis(bc = bcs)
            gphi = space1.grad_basis(bc = bcs)
            cm = mesh0.entity_measure('cell')
            M = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi ,cm)  
            P = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)
        else:
            cell2dof = space.cell_to_dof()
            GDOF = space.number_of_global_dofs()
            phi = space.basis(bcs)
            gphi = space.grad_basis(bcs)
            cm = self.cm
            M = bm.einsum('q , cqi ,cqj, c -> cij ',ws, phi ,phi ,cm)  
            P = bm.einsum('q , cqid , cid ,cqj ,c -> cij' , ws , gphi, delta_x[cell2dof], phi, cm)

        I = bm.broadcast_to(cell2dof[:, :, None], shape=P.shape)
        J = bm.broadcast_to(cell2dof[:, None, :], shape=P.shape)
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
        P = csr_matrix((P.flat, (I.flat, J.flat)), shape=(GDOF, GDOF))
        # ml = pyamg.ruge_stuben_solver(M)
        def ODEs(t,y):
            f = spsolve1(M, P @ y)
            return f
        # 初值条件  
        uh0 = self.uh
        # 范围
        tau_span = [0,1]
        # 求解
        sol = solve_ivp(ODEs,tau_span,uh0,method='RK23').y[:,-1]
        return sol
    
    def construct(self,new_node):
        """
        @brief construct: 重构信息
        @param new_mesh:新的节点
        """
        # node 更新之前完成插值
        self.uh = self.interpolate(new_node)
        self.mesh.node = new_node
        self.space.mesh = self.mesh
        # uh = self.space.function(self.uh)
        # error = self.mesh.error(uh,0)
        # print(f'范数为{error}')
        self.node = new_node
        self.cm = self.mesh.entity_measure('cell')
        self.star_measure = self.get_star_measure()
        self.G,self.M = self.get_control_function()

    def construct_with_pde(self,new_node,pde):
        self.mesh.node = new_node
        self.space.mesh = self.mesh

        # bform = BilinearForm(self.space)
        # lform = LinearForm(self.space)

        # SDI = ScalarDiffusionIntegrator(q = self.p+1,method='isopara')
        # SSI = ScalarSourceIntegrator(source = pde.source,method='isopara')
        # bform.add_integrator(SDI)
        # lform.add_integrator(SSI)
        # A = bform.assembly()
        # b = lform.assembly()

        # bc = DirichletBC(self.space, gd = lambda p : pde.dirichlet(p))
        # A ,b = bc.apply(A,b)
        # self.uh = spsolve(A,b,solver='scipy')
        self.uh = pde.solution(new_node)

        self.node = new_node
        self.cm = self.mesh.entity_measure('cell')
        self.star_measure = self.get_star_measure()
        self.G,self.M = self.get_control_function()

    def preprocessor(self,uh0,pde = None, steps = 10):
        """
        @brief preprocessor: 预处理器
        @param steps: 伪时间步数
        """
        self.uh = 1/steps * uh0
        self.G,self.M = self.get_control_function()
        for i in range(steps):
            t = (i+1)/steps
            self.uh = t * uh0
            self.mesh , self.uh = self.mesh_redistribution(self.uh,pde = pde)
            # self.mesh , self.uh = self.mesh_redistribution(self.uh)
            uh0 = self.uh
        return self.mesh , self.uh
            

    def caculate_tol(self):
        """
        @brief caculate_tol: 计算容许误差
        """
        logic_mesh = self.logic_mesh
        logic_cm = logic_mesh.entity_measure('cell')
        logic_em = logic_mesh.entity_measure('edge')
        cell2edge = logic_mesh.cell_to_edge()
        em_cell = logic_em[cell2edge]
        if self.TD == 3:
            mul = em_cell[:,:3]*em_cell[:,3:][:,::-1]
            p = 0.5*bm.sum(mul,axis=1)
            d = bm.min(bm.sqrt(p*(p-mul[:,0])*(p-mul[:,1])*(p-mul[:,2]))/(3*logic_cm))
        else:
            d = bm.min(bm.prod(em_cell,axis=1)/(2*logic_cm))
        return d*0.1/self.p
    
    def mesh_redistribution(self ,uh, tol = None , pde = None , maxit = 1000):
        """
        @brief mesh_redistribution: 网格重构算法
        @param tol: 容许误差
        @param maxit 最大迭代次数
        """
        import matplotlib.pyplot as plt
        self.uh = uh
        if tol is None:
            tol = self.tol
            print(f'容许误差为{tol}')

        for i in range(maxit):
            logic_node,vector_field = self.solve_move_LogicNode()
            
            L_infty_error = bm.max(bm.linalg.norm(self.logic_node - logic_node,axis=1))
            print(f'第{i+1}次迭代的差值为{L_infty_error}')
            if L_infty_error < tol:
                print(f'迭代总次数:{i+1}次')
                return self.mesh , self.uh
            elif i == maxit - 1:
                print('超出最大迭代次数')
                break
            node = self.get_physical_node(vector_field,logic_node)
            if pde is not None:
                self.construct_with_pde(node,pde)
            else:
                self.construct(node)




class Mesh_Data_Harmap():
    def __init__(self,mesh:Union[TriangleMesh,
                                 TetrahedronMesh,
                                 LagrangeTriangleMesh],
                     Vertex) -> None:
        self.isinstance_mesh_type(mesh)
        if self.mesh_type == "LagrangeTriangleMesh":
            self.mesh = mesh.linearmesh
        else:
            self.mesh = mesh
        self.node = mesh.entity('node')
        self.isBdNode = mesh.boundary_node_flag()
        self.Vertex = Vertex
        self.isconvex = self.is_convex()
        
    def isinstance_mesh_type(self,mesh):
        if isinstance(mesh, TriangleMesh):
            self.mesh_type = "TriangleMesh"
        elif isinstance(mesh, TetrahedronMesh):
            self.mesh_type = "TetrahedronMesh"
        elif isinstance(mesh, LagrangeTriangleMesh):
            self.mesh_type = "LagrangeTriangleMesh"
            self.Lmesh = mesh
        else:
            raise TypeError("Unsupported mesh type")

    def is_convex(self):
        """
        判断边界是否是凸的
        """
        from scipy.spatial import ConvexHull
        Vertex = self.Vertex
        hull = ConvexHull(Vertex)
        return len(Vertex) == len(hull.vertices)
    
    def sort_bdnode_and_bdface(self) -> TensorLike:
        mesh = self.mesh
        BdNodeidx = mesh.boundary_node_index()
        BdEdgeidx = mesh.boundary_face_index()
        node = mesh.node
        edge = mesh.edge
        cell = mesh.cell
        mesh_0 = TM(bm.to_numpy(node),bm.to_numpy(cell))
        node2edge = mesh_0.ds.node_to_edge()
        bdnode2edge = node2edge[BdNodeidx][:,BdEdgeidx]
        i,j = bm.nonzero(bm.tensor(bdnode2edge.todense()))
        bdnode2edge = j.reshape(-1,2)
        glob_bdnode2edge = bm.zeros_like(node,dtype=bm.int64)
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
        sort_glob_bdnode_idx = bm.array(sort_glob_bdnode_idx_list,dtype=bm.int32)
        sort_glob_bdedge_idx = bm.array(sort_glob_bdedge_idx_list,dtype=bm.int32)
        if self.mesh_type == "LagrangeTriangleMesh":
            Lmesh = self.Lmesh
            Ledge = Lmesh.edge
            sort_glob_bdedge = Ledge[sort_glob_bdedge_idx]
            sort_glob_bdnode_idx = sort_glob_bdedge[:,:2].flatten()

        return sort_glob_bdnode_idx,sort_glob_bdedge_idx
    
    def get_normal_inform(self,sort_BdNode_idx = None) -> None:
        mesh = self.mesh
        BdNodeidx = mesh.boundary_node_index()
        if sort_BdNode_idx is not None:
            BdNodeidx = sort_BdNode_idx
        BdFaceidx = mesh.boundary_face_index()
        TD = mesh.top_dimension()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        if TD == 2:
            mesh1 = TM(bm.to_numpy(node),bm.to_numpy(cell))
            node2face = mesh1.ds.node_to_face()
        elif TD == 3:
            def node_to_face(mesh): # 作为三维网格的辅助函数
                NN = mesh.number_of_nodes()
                NF = mesh.number_of_faces()
                face = mesh.entity('face')
                NVF = 3
                node2face = csr_matrix(
                        (
                            bm.ones(NVF*NF, dtype=bm.bool),
                            (
                                face.flatten(),
                                bm.repeat(bm.arange(NF), NVF)
                            )
                        ), shape=(NN, NF))
                return node2face
            node2face = node_to_face(mesh)
        bd_node2face = node2face[BdNodeidx][:,BdFaceidx]
        i , j = bm.nonzero(bm.tensor(bd_node2face.todense()))
        bdfun = mesh.face_unit_normal(index=BdFaceidx[j])
        tolerance = 1e-8
        bdfun_rounded = bm.round(bdfun / tolerance) * tolerance
        normal,inverse = bm.unique(bdfun_rounded,return_inverse=True ,axis = 0)
        _,index,counts = bm.unique(i,return_index=True,return_counts=True)
        cow = bm.max(counts)
        r = bm.min(counts)

        node2face_normal = -bm.ones((BdNodeidx.shape[0],cow),dtype=bm.int64)
        node2face_normal = bm.set_at(node2face_normal,(slice(None),slice(r)),inverse[index[:,None]+bm.arange(r)])
        for i in range(cow-r):
            isaimnode = counts > r+i
            node2face_normal = bm.set_at(node2face_normal,(isaimnode,r+i) ,
                                            inverse[index[isaimnode]+r+i])
        
        for i in range(node2face_normal.shape[0]):
            x = node2face_normal[i]
            unique_vals = bm.unique(x[x >= 0])
            result = -bm.ones(TD, dtype=bm.int32)
            result[:len(unique_vals)] = unique_vals
            node2face_normal[i,:TD] = result

        return node2face_normal[:,:TD],normal
    
    def get_basic_infom(self):
        mesh = self.mesh
        node2face_normal,normal = self.get_normal_inform()
        BdNodeidx = mesh.boundary_node_index()
        Bdinnernode_idx = BdNodeidx[node2face_normal[:,1] < 0]
        if self.mesh_type == "LagrangeTriangleMesh":
            Lmesh = self.Lmesh
            BdFaceidx = Lmesh.boundary_face_index()
            LBdedge = Lmesh.edge[BdFaceidx]
            Bdinnernode_idx = bm.concatenate([Bdinnernode_idx,LBdedge[:,1:-1].flatten()])
        is_convex = self.isconvex
        Arrisnode_idx = None
        if is_convex:
            Vertex_idx = BdNodeidx[node2face_normal[:,-1] >= 0]
            if mesh.TD == 3:
                Arrisnode_idx = BdNodeidx[(node2face_normal[:,1] >= 0) & (node2face_normal[:,-1] < 0)]
            return Vertex_idx,Bdinnernode_idx,Arrisnode_idx
        else:
            if self.Vertex is None:
                raise ValueError('The boundary is not convex, you must give the Vertex')
            minus = mesh.node - self.Vertex[:,None]
            judge_vertex = bm.sum(((minus**2)[:,:,0],(minus**2)[:,:,1]),axis=0) < 1e-10
            K = bm.arange(mesh.number_of_nodes())
            Vertex_idx = judge_vertex @ K
            sort_Bdnode_idx,sort_Bdface_idx = self.sort_bdnode_and_bdface()
            return Vertex_idx,Bdinnernode_idx,sort_Bdnode_idx