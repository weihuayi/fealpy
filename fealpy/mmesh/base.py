from .config import *
from .tool import _solve_quad_parametric_coords,_solve_hex_parametric_coords
from ..decorator import variantmethod

class MM_PREProcessor:
    def __init__(self,mesh:_U,space:_V,config:Config) -> None:
        self.mesh = mesh
        self.pspace = space
        self.vertices = mesh.nodedata['vertices']
        self.method = config.active_method
        self.logic_domain = config.logic_domain
        self.int_meth = config.int_meth
        self.mol_meth = config.mol_meth

        self.prepare()

    def prepare(self):
        """
        prepare the basic information
        """
        if self.method == 'Harmap':
            self._data_and_device()
            self._isinstance_mesh_type()
            self._meshtop_preparation()
            self._geometry_preparation()
            self._space_preparation()
        elif self.method in ['PSMFEM','HousMMPDE']:
            self._data_and_device()
            self._isinstance_mesh_type()
            logic_node = bm.copy(self.mesh.node)
            logic_cell = bm.copy(self.mesh.cell)
            if self.mesh_type =="LagrangeTriangleMesh":
                self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh,self.p)
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh,self.p)
            else:
                self.logic_mesh = self.mesh_class(logic_node,logic_cell)
            self.logic_cm = self.logic_mesh.entity_measure('cell')
            self._meshtop_preparation()
            self._geometry_preparation()
            self._space_preparation()
            self._logic_space_preparation()
            
        elif self.method == 'GFMMPDE':
            self._data_and_device()
            self._isinstance_mesh_type()
            logic_node = bm.copy(self.mesh.node)
            logic_cell = bm.copy(self.mesh.cell)
            if self.mesh_type =="LagrangeTriangleMesh":
                self.logic_mesh = self.mesh_class.from_triangle_mesh(self.linermesh,self.p)
            elif self.mesh_type == "LagrangeQuadrangleMesh":
                self.logic_mesh = self.mesh_class.from_quadrangle_mesh(self.linermesh,self.p)
            else:
                self.logic_mesh = self.mesh_class(logic_node,logic_cell)
            self.logic_cm = self.logic_mesh.entity_measure('cell')
            self._meshtop_preparation()
            if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
                p_mesh = self.linermesh
                self.lmspace = ParametricLagrangeFESpace(self.logic_mesh, p=self.p)
            else:
                p_mesh = self.mesh
                self.lmspace = LagrangeFESpace(self.logic_mesh, p=self.p)
            if self.mesh.TD == 2:
                (self.Vertices_idx,
                self.Bdinnernode_idx,
                self.sort_BdNode_idx) = self._get_various_bdidx(p_mesh,True)
                self._align_boundary_with_vertices()
            else:
                self.isconvex = self._is_convex()
                (self.Vertices_idx,
                self.Bdinnernode_idx,
                self.Arrisnode_idx) = self._get_various_bdidx(p_mesh,False)
                self.arris2node = self._arris_to_node(self.mesh)
                self.Bi_Pnode_normal,self.Ar_Pnode_normal,bcollection = self._get_normal_information(self.mesh)
                self.b_val0 = bcollection[0]
            self._space_preparation()

    def _meshtop_preparation(self):
        """
        save the mesh topology information
        """
        mesh = self.mesh
        self.node = mesh.entity('node')
        self.cell = mesh.entity('cell')
            
        self.node2face = self.mesh.node_to_face().tocsr()
        
        self.isBdNode = mesh.boundary_node_flag()
        self.BdNodeidx = mesh.boundary_node_index()
        self.BdFaceidx = mesh.boundary_face_index()

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()
        self.NN = mesh.number_of_nodes()
        self.NC = mesh.number_of_cells()
        self.BDNN = len(self.BdNodeidx)

        self.cm = mesh.entity_measure('cell')
        self.rm = mesh.reference_cell_measure()
        if self.mesh_type == "LagrangeQuadrangleMesh":
            self.hmin = bm.sqrt(bm.min(mesh.entity_measure('cell')))
        else:
            self.hmin = bm.min(mesh.entity_measure('edge'))
        
        if self.mesh_type == "TriangleMesh":
            self.Jacobi = mesh.jacobian_matrix
        else:
            self.Jacobi = mesh.jacobi_matrix

    def _data_and_device(self):
        """
        save the data and device information
        """
        node = self.mesh.node
        cell = self.mesh.cell
        self.kwargs0 = bm.context(node)
        self.kwargs1 = bm.context(cell)
        self.itype = cell.dtype
        self.ftype = node.dtype
        self.device = self.mesh.device
        if self.device == 'cuda':
            self.solver = 'cupy'
        else:
            self.solver = 'scipy'  # or 'mumps'

    def _isinstance_mesh_type(self):
        """
        Check the mesh type and set the mesh information
        """
        mesh = self.mesh
        mesh_mapping = {
        TriangleMesh: ("TriangleMesh","Simplexmesh", None, 1),
        TetrahedronMesh: ("TetrahedronMesh","Simplexmesh", None, 1),
        LagrangeTriangleMesh: ("LagrangeTriangleMesh","Simplexmesh", "isopara", None),
        QuadrangleMesh: ("QuadrangleMesh", "Tensormesh", "isopara", 1),
        HexahedronMesh: ("HexahedronMesh",  "Tensormesh", "isopara", 1),
        LagrangeQuadrangleMesh: ("LagrangeQuadrangleMesh", "Tensormesh", "isopara", None)
        }

        for mesh_class, (mesh_type, g_type, assambly_method,p) in mesh_mapping.items():
            if isinstance(mesh, mesh_class):
                self.mesh_type = mesh_type
                self.mesh_class = mesh_class
                self.g_type = g_type
                self.assambly_method = assambly_method
                self.p = getattr(mesh, 'p', 1)
                if mesh_type in ["LagrangeTriangleMesh", "LagrangeQuadrangleMesh"]:
                    self.linermesh = mesh.linearmesh
                break

    def _geometry_preparation(self):
        """
        get the geometry information
        """
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            p_mesh = self.linermesh
        else:
            p_mesh = self.mesh
        self.isconvex = self._is_convex()
        if self.isconvex:
            (self.Vertices_idx,
             self.Bdinnernode_idx,
             self.Arrisnode_idx) = self._get_various_bdidx(p_mesh,False)
        else:
            (self.Vertices_idx,
             self.Bdinnernode_idx,
             self.sort_BdNode_idx) = self._get_various_bdidx(p_mesh,True)
            "Align boundary points with vertices"
            if self.sort_BdNode_idx[0] != self.Vertices_idx[0]:
                K = bm.where(self.sort_BdNode_idx == self.Vertices_idx[0])[0][0]
                self.sort_BdNode_idx = bm.roll(self.sort_BdNode_idx,-K)
        
        if self.TD == 2:
            if self.isconvex:
                self.Bi_Pnode_normal,self.b_val0 = self._get_normal_information(self.mesh)
                self.Bi_Lnode_normal = self.Bi_Pnode_normal
                self.logic_Vertices = self.node[self.Vertices_idx]
            else:
                self.Bi_Pnode_normal = self._get_normal_information(self.mesh)
                self.Bi_Lnode_normal,self.b_val0,self.logic_Vertices = self._get_logic_boundary()
        else:
            self.Bi_Lnode_normal,self.Ar_Lnode_normal,bcollection = self._get_normal_information(self.mesh)
            self.Bi_Pnode_normal = self.Bi_Lnode_normal
            self.logic_Vertices = self.node[self.Vertices_idx]
            self.b_val0 = bcollection[0]
            self.b_val1 = bcollection[1]
            self.b_val2 = bcollection[2]

    def _logic_space_preparation(self):
        """
        prepare the logic space information
        """
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            self.lmspace = ParametricLagrangeFESpace(self.logic_mesh, p=self.p)
        else:
            self.lmspace = LagrangeFESpace(self.logic_mesh, p=self.p)
        SMI = ScalarMassIntegrator(q= self.q , method=self.assambly_method)
        bform = BilinearForm(self.lmspace)
        bform.add_integrator(SMI)
        self.logic_mass = bform.assembly()

    def _space_preparation(self):
        """
        get the space information
        """
        if self.int_meth == 'comass':
            self.q = self.pspace.p + 1
        else:
            self.q = self.p + 1

        qf = self.mesh.quadrature_formula(self.q)
        self.bcs, self.ws = qf.get_quadrature_points_and_weights()
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            self.mspace = ParametricLagrangeFESpace(self.mesh, p=self.p)
        else:
            self.mspace = LagrangeFESpace(self.mesh, p=self.p)
        self.d = self._sqrt_det_G(self.bcs)
        self.sm = self._get_star_measure()
        self.cell2dof = self.mspace.cell_to_dof()

        if self.g_type == "Tensormesh":
            ml = bm.multi_index_matrix(self.p,1,dtype=self.ftype)/self.p
            self.multi_index = tuple(ml for _ in range(self.TD))
        else:
            self.multi_index = bm.multi_index_matrix(self.p,self.TD,dtype=self.ftype)/self.p

        NLI = self.mesh.number_of_local_ipoints(self.pspace.p)
        shape = (self.NC,NLI,NLI)
        self.GDOF = self.mspace.number_of_global_dofs()
        self.pcell2dof = self.pspace.cell_to_dof()
        self.I = bm.broadcast_to(self.pcell2dof[:, :, None], shape=shape)
        self.J = bm.broadcast_to(self.pcell2dof[:, None, :], shape=shape)

        self.SMI = ScalarMassIntegrator(q= self.q , method=self.assambly_method)
        self.SDI = ScalarDiffusionIntegrator(q= self.q , method=self.assambly_method)
        self.SSI = ScalarSourceIntegrator(q= self.q,method=self.assambly_method)

        if self.int_meth == 'comass' or self.mol_meth == 'heatequ':
            print("Using mass matrix as update matrix")
            self.update_matrix = self._mass_gererator
        else:
            self.update_matrix = lambda: None
        self.update_matrix()

    def _is_convex(self) -> bool:
        """
        judge the mesh is convex or not
        
        Returns
            bool: True if mesh is convex, False otherwise
        """
        from scipy.spatial import ConvexHull
        vertices = self.vertices
        if isinstance(vertices, list):
            vertices = bm.concat(vertices, axis=0)
        hull = ConvexHull(vertices)
        return len(vertices) == len(hull.vertices)
    
    def _sqrt_det_G(self,bcs)->TensorLike:
        """
        calculate the square root of the determinant of the first fundamental form
        
        Parameters
            bcs: TensorLike, barycentric coordinates
        Returns
            TensorLike: square root of determinant
        """
        J = self.Jacobi(bcs)
        ndim = J.ndim
        axes = list(range(ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        G = bm.permute_dims(J,axes=axes) @ J
        d = bm.sqrt(bm.linalg.det(G))
        d = d[(...,)+ (2 - d.ndim) * (None,)]
        return d

    def _align_group_boundary(self, group_boundary, vertices_group):
        """
        Align a single group's boundary with its vertices
        
        Parameters
            group_boundary: TensorLike, boundary nodes of a group
            vertices_group: TensorLike, vertices of the group
        Returns
            TensorLike: aligned boundary nodes
        """
        # 检查第一个顶点是否已经在边界的起始位置
        if group_boundary[0] != vertices_group[0]:
            # 找到第一个顶点在边界中的位置
            vertex_positions = []
            for vertex in vertices_group:
                pos = bm.where(group_boundary == vertex)[0]
                if len(pos) > 0:
                    vertex_positions.append(pos[0])
            
            if len(vertex_positions) > 0:
                # 使用第一个找到的顶点作为起始点
                K = vertex_positions[0]
                group_boundary = bm.roll(group_boundary, -K)
        return group_boundary
    
    def _align_boundary_with_vertices(self):
        """
        统一处理边界对齐，支持单连通和多连通区域
        """
        # 确保 Vertices_idx 和 circle_id 都是列表格式
        vertices_groups = self.Vertices_idx if isinstance(self.Vertices_idx, list) else [self.Vertices_idx]
        circle_ids = getattr(self, 'circle_id', [0])  # 如果没有 circle_id，默认从0开始
        
        aligned_segments = []
        
        for group_idx, vertices_group in enumerate(vertices_groups):
            # 获取当前组的边界节点段
            start_idx = circle_ids[group_idx]
            end_idx = circle_ids[group_idx + 1] if group_idx + 1 < len(circle_ids) else len(self.sort_BdNode_idx)
            group_boundary = self.sort_BdNode_idx[start_idx:end_idx]
            
            # 对当前组进行边界对齐
            aligned_group = self._align_group_boundary(group_boundary, vertices_group)
            aligned_segments.append(aligned_group)
        
        # 重新组合对齐后的边界节点
        self.sort_BdNode_idx = bm.concat(aligned_segments, axis=0)


    def _mass_gererator(self):
        """
        generate the mass matrix
        """
        pspace = self.pspace
        SMI = self.SMI
        bform = BilinearForm(pspace)
        bform.add_integrator(SMI)
        A = bform.assembly()
        self.mass = A

    def _get_star_measure(self)->TensorLike:
        """
        get the measure of the star shape
        
        Returns
            TensorLike: star measure values
        """
        NN = self.NN
        d = self.d
        rm = self.rm
        star_measure = bm.zeros(NN,**self.kwargs0)
        cell_areas = bm.sum(rm*d* self.ws[None,:], axis=1)
        bm.index_add(star_measure , self.cell , cell_areas[:,None])
        return star_measure
    
    def _sort_bdnode_and_bdface(self,mesh:_U) -> TensorLike:
        """
        sort the boundary node and boundary face
        
        Parameters
            mesh: _U, mesh instance of process mesh(pmesh)
        Returns
            TensorLike: sorted boundary nodes and faces
        """
        BdNodeidx = bm.asarray(mesh.boundary_node_index(),**self.kwargs1)
        BdEdgeidx = bm.asarray(mesh.boundary_face_index(),**self.kwargs1)
        node = mesh.node
        edge = mesh.edge
        
        node2edge = mesh.node_to_edge().tocsr()
        NBB = node2edge[BdNodeidx][:,BdEdgeidx]
        i = NBB.row
        j = NBB.col

        bdnode2edge = j.reshape(-1,2)
        glob_bdnode2edge = bm.zeros_like(node,**self.kwargs1)
        glob_bdnode2edge = bm.set_at(glob_bdnode2edge,BdNodeidx,BdEdgeidx[bdnode2edge])
        
        sort_glob_bdedge_idx_list = []
        sort_glob_bdnode_idx_list = []

        start_bdnode_idx = BdNodeidx[0]
        sort_glob_bdnode_idx_list.append(start_bdnode_idx)
        current_node_idx = start_bdnode_idx
        self.circle_id = [0]
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
                    self.circle_id.append(i+1) # 记录闭环的起始位置
                    remian_bdnode_idx = list(set(BdNodeidx)-set(sort_glob_bdnode_idx_list))
                    start_bdnode_idx = remian_bdnode_idx[0]
                    next_node_idx = start_bdnode_idx
                else:
                # 闭环跳出循环
                    break
            sort_glob_bdnode_idx_list.append(next_node_idx)
            current_node_idx = next_node_idx
        sort_glob_bdnode_idx = bm.array(sort_glob_bdnode_idx_list,**self.kwargs1)
        sort_glob_bdedge_idx = bm.array(sort_glob_bdedge_idx_list,**self.kwargs1)
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            Ledge = self.mesh.edge
            sort_glob_bdedge = Ledge[sort_glob_bdedge_idx]
            sort_glob_bdnode_idx = sort_glob_bdedge[:,:-1].flatten()

        return sort_glob_bdnode_idx,sort_glob_bdedge_idx
    
    def _get_node2face_norm(self,mesh:_U) -> TensorLike:
        """
        get node to face normal
        
        Parameters
            mesh: _U, mesh instance
        Returns
            TensorLike: node to face normal mapping
        """
        BdNodeidx = mesh.boundary_node_index()
        BdFaceidx = mesh.boundary_face_index()
        TD = mesh.top_dimension()
        node2face = self.node2face
        NBB = node2face[BdNodeidx][:,BdFaceidx]
        i = NBB.row
        j = NBB.col

        bdfun = mesh.face_unit_normal(index=BdFaceidx[j])
        tolerance = 1e-8
        bdfun_rounded = bm.round(bdfun / tolerance) * tolerance
        normal,inverse = bm.unique(bdfun_rounded,return_inverse=True ,axis = 0)
        _,index,counts = bm.unique(i,return_index=True,return_counts=True)
        cow = bm.max(counts)
        r = bm.min(counts)

        inverse = bm.asarray(inverse,**self.kwargs1)
        node2face_normal = -bm.ones((BdNodeidx.shape[0],cow),**self.kwargs1)
        node2face_normal = bm.set_at(node2face_normal,(slice(None),slice(r)),
                                     inverse[index[:,None]+bm.arange(r ,**self.kwargs1)])
        for i in range(cow-r):
            isaimnode = counts > r+i
            node2face_normal = bm.set_at(node2face_normal,(isaimnode,r+i) ,
                                            inverse[index[isaimnode]+r+i])
        
        for i in range(node2face_normal.shape[0]):
            x = node2face_normal[i]
            unique_vals = bm.unique(x[x >= 0])
            result = -bm.ones(TD, **self.kwargs1)
            result = bm.set_at(result, slice(len(unique_vals)), unique_vals)
            node2face_normal = bm.set_at(node2face_normal, (i,slice(TD)) , result)

        return node2face_normal[:,:TD],normal
    
    def _get_various_bdidx(self,mesh:_U,is_bdsort:bool) -> TensorLike:
        """
        get various boundary index
        
        Parameters
            mesh: _U, mesh instance
            is_bdsort: bool, whether to sort boundary
        Returns
            TensorLike: various boundary indices
        """
        node2face_normal,normal = self._get_node2face_norm(mesh)
        BdNodeidx = mesh.boundary_node_index()
        Bdinnernode_idx = BdNodeidx[node2face_normal[:,1] < 0]
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            BdFaceidx = self.BdFaceidx
            LBdedge = self.mesh.edge[BdFaceidx]
            Bdinnernode_idx = bm.concat([Bdinnernode_idx,LBdedge[:,1:-1].flatten()])

        Arrisnode_idx = None
        if not is_bdsort:
            Vertices_idx = BdNodeidx[node2face_normal[:,-1] >= 0]
            if mesh.TD == 3:
                second_vol_idx = node2face_normal[:,1] >= 0
                # 找出棱内点
                Arrisnode_idx = BdNodeidx[(second_vol_idx) & (node2face_normal[:,-1] < 0)]
            return Vertices_idx,Bdinnernode_idx,Arrisnode_idx
        else:
            if self.vertices is None:
                raise ValueError('The boundary is not convex, you must give the Vertices')
            if isinstance(self.vertices, list): # 需要处理多连通情况
                Vertices_idx = []
                for vertices_group in self.vertices:
                    minus = mesh.node - vertices_group[:, None]
                    judge_Vertices = bm.array(bm.sum(minus**2, axis=-1) < 1e-10, **self.kwargs1)
                    K = bm.arange(mesh.number_of_nodes(), **self.kwargs1)
                    Vertices_idx_group = bm.matmul(judge_Vertices, K)
                    Vertices_idx.append(Vertices_idx_group)
            else:
                minus = mesh.node - self.vertices[:, None]
                judge_Vertices = bm.array(bm.sum(minus**2, axis=-1) < 1e-10, **self.kwargs1)
                K = bm.arange(mesh.number_of_nodes(), **self.kwargs1)
                Vertices_idx = bm.matmul(judge_Vertices, K)
            sort_Bdnode_idx,sort_Bdface_idx = self._sort_bdnode_and_bdface(mesh)
            return Vertices_idx,Bdinnernode_idx,sort_Bdnode_idx

    def _arris_to_node(self,mesh:_U):
        from ..sparse import coo_matrix
        node2face_normal,normal = self._get_node2face_norm(mesh)
        bd_arris_idx = (node2face_normal[:,1] >= 0) & (node2face_normal[:,-1] < 0)
        # 给棱上点进行逐棱分类
        Arris_sort_vol = bm.sort(node2face_normal[bd_arris_idx,:-1],axis=-1)
        t,inverse = bm.unique(Arris_sort_vol,return_inverse=True,axis=0)
        # row 表示内点对应的棱编号
        row = inverse 
        col = self.Arrisnode_idx
        # 使用棱点对角点的关系，找到每个棱内点的邻接角点
        node2node = self.mesh.node_to_node(format='csr')
        shift = node2node[self.Arrisnode_idx, self.Vertices_idx]
        i,j = shift.row, shift.col
        # 按照棱内点的顺序将所有的邻接角点添加到对应的角点索引中
        row_exp = bm.concat([row, row[i]],axis=0)
        col_exp = bm.concat([col, self.Vertices_idx[j]],axis=0)
        # 将其存储为稀疏布尔格式
        value = bm.ones_like(row_exp,dtype=bm.bool, device=self.device)
        arris_to_node = coo_matrix((value, (row_exp, col_exp)), shape=(len(t), self.NN))
        return arris_to_node.tocsr()
    
    def _vertice_and_arris(self,mesh:_U):
        BdNodeidx = self.BdNodeidx
        node2face_normal,normal = self._get_node2face_norm(mesh)
        vertice_arris_idx = (node2face_normal[:,1] >= 0)
        return BdNodeidx[vertice_arris_idx]
        
    def _arris_to_vertice(self,mesh:_U):
        pass

    def _vertice_to_arris(self,mesh:_U):
        pass

    def _surface_to_vertice(self,mesh:_U):
        pass
    
    def _vertice_to_surface(self,mesh:_U):
        pass

    def _arris_to_surface(self,mesh:_U):
        pass

    def _surface_to_arris(self,mesh:_U):
        pass

    def _get_normal_information(self,mesh:_U) -> TensorLike:
        """
        get the normal information
        
        Parameters
            mesh: _U, mesh instance
        Returns
            TensorLike: normal information
        """
        Bdinnernode_idx = self.Bdinnernode_idx
        BdFaceidx = self.BdFaceidx
        node2face = self.node2face
        if self.TD == 3:
            Arrisnode_idx = self.Arrisnode_idx
            NAB = node2face[Arrisnode_idx][:,BdFaceidx]
            i0 = NAB.row
            j0 = NAB.col
            bdfun0 = mesh.face_unit_normal(index=BdFaceidx[j0])
            normal0,inverse0 = bm.unique(bdfun0,return_inverse=True ,axis = 0)
            _,index0,counts0 = bm.unique(i0,return_index=True,return_counts=True)   
            maxcount = bm.max(counts0)
            mincount = bm.min(counts0)
            Ar_node2normal_idx = -bm.ones((len(Arrisnode_idx),maxcount),**self.kwargs1)
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
                Ar_node2normal_idx = bm.set_at(Ar_node2normal_idx,(i,slice(len(unique_vals))),unique_vals)
            Ar_node2normal = normal0[Ar_node2normal_idx[:,:2]]  
            
        if self.mesh_type in ["LagrangeTriangleMesh" , "LagrangeQuadrangleMesh"]:
            LBdFace = mesh.face[BdFaceidx]
            LBd_node2face = bm.zeros((self.NN , 2),  **self.kwargs1)
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,:-1],0) , BdFaceidx[:,None])
            LBd_node2face = bm.set_at(LBd_node2face , (LBdFace[:,1:],1) , BdFaceidx[:,None])
            LBdi_node2face = LBd_node2face[Bdinnernode_idx]
            linear_mesh = mesh.linearmesh
            Bi_node_normal  = linear_mesh.face_unit_normal(index=LBdi_node2face[:,0])
        else:
            NBB = node2face[Bdinnernode_idx][:,BdFaceidx]
            i1 = NBB.row
            j1 = NBB.col
            bdfun1 = mesh.face_unit_normal(index=BdFaceidx[j1])
            _,index1 = bm.unique(i1,return_index=True)
            Bi_node_normal = bdfun1[index1]
        
        if not self.isconvex:
            return Bi_node_normal
        else:
            Binode = self.node[Bdinnernode_idx]
            b_val0 = bm.sum(Binode * Bi_node_normal,axis=1)
            if self.TD == 2:
                return Bi_node_normal, b_val0
            else:
                Ar_node = self.node[Arrisnode_idx]
                b_val1 = bm.sum(Ar_node*Ar_node2normal[:,0,:],axis=1)
                b_val2 = bm.sum(Ar_node*Ar_node2normal[:,1,:],axis=1)
                return Bi_node_normal, Ar_node2normal, (b_val0, b_val1, b_val2)
            
    def _logic_domain_generator(self,initial_logic_domain,physics_domain) -> TensorLike:
        """
        generate the logic domain
        
        Parameters
            initial_logic_domain: TensorLike, initial guess of the logic domain
            physics_domain: TensorLike, physics domain
        Returns
            TensorLike: generated logic domain
        """
        from scipy.optimize import minimize
        num_sides = physics_domain.shape[0]
        edge_lengths = bm.sqrt(bm.sum((physics_domain - bm.roll(physics_domain, -1, axis=0))**2, axis=1))
        kwargs0 = self.kwargs0
        def objective_function(vertices):
            vertices = bm.asarray(vertices.reshape((num_sides, 2)), **kwargs0)
            logic_edge_lengths = bm.sqrt(bm.sum((vertices - bm.roll(vertices, -1, axis=0))**2, axis=1))
            return bm.sum((logic_edge_lengths - edge_lengths)**2)

        def convexity_constraint(vertices):
            vertices = vertices.reshape((num_sides, 2))
            angles = []
            for i in range(num_sides):
                v1 = vertices[(i + 1) % num_sides] - vertices[i]
                v2 = vertices[(i - 1) % num_sides] - vertices[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                angles.append(cross)
            return bm.array(angles, **kwargs0)  # 确保所有叉积符号相同（凸多边形）

        initial_guess = initial_logic_domain.flatten()
        constraints = {'type': 'ineq', 'fun': convexity_constraint}
        result = minimize(objective_function, initial_guess, constraints=constraints, method='SLSQP')

        logic_domain = bm.asarray(result.x, **kwargs0).reshape((num_sides, 2))
        return logic_domain
    
    def _get_logic_boundary(self,p = None) -> TensorLike:
        """
        get the logic boundary
        
        Parameters
            p: optional, polynomial degree
        Returns
            TensorLike: logic boundary information
        """
        sBdNodeidx = self.sort_BdNode_idx
        Bdinnernode_idx = self.Bdinnernode_idx
        logic_domain = self.logic_domain
        node = self.node
        Verticesidx = self.Vertices_idx
        physics_domain = node[Verticesidx]
        num_sides = physics_domain.shape[0]
        if logic_domain is None:
            angles = bm.linspace(0,2*(1-1/(num_sides))*bm.pi,num_sides,**self.kwargs0)
            initial_logic_domain = bm.stack([bm.cos(angles),bm.sin(angles)],axis=1)
            # generate logic domain
            logic_domain = self._logic_domain_generator(initial_logic_domain,physics_domain)

        Lside_vector = bm.roll(logic_domain,-1,axis=0) - logic_domain
        Lside_vector_rotated = bm.stack([Lside_vector[:, 1], -Lside_vector[:, 0]], axis=1)
        Lside_length = bm.linalg.norm(Lside_vector_rotated, axis=1)
        Logic_unit_norm = Lside_vector_rotated / Lside_length[:, None]
        b_part = bm.sum(Logic_unit_norm * logic_domain,axis=1)
        K = bm.where(sBdNodeidx[:,None] == Verticesidx)[0]
        K = bm.concat([K,bm.array([len(sBdNodeidx)])])
        if p is None:
            Lun_repeat = bm.repeat(Logic_unit_norm,K[1:]-K[:-1],axis=0)
            bp_repeat = bm.repeat(b_part,K[1:]-K[:-1],axis=0)
            LBd_node2unorm = bm.zeros((self.NN , 2),  **self.kwargs0)
            b_val = bm.zeros(self.NN,  **self.kwargs0)
            bm.index_add(LBd_node2unorm , sBdNodeidx , Lun_repeat)
            bm.index_add(b_val , sBdNodeidx , bp_repeat)
            LBd_node2unorm = LBd_node2unorm[Bdinnernode_idx]
            b_val = b_val[Bdinnernode_idx]
            
            return LBd_node2unorm,b_val,logic_domain
        else:
            if self.mesh_type in ["LagrangeTriangleMesh","LagrangeQuadrangleMesh"]:
                node = self.linermesh.node
                map = bm.where(self.sort_BdNodeidx < len(node))[0]
                sBdNodeidx = self.sort_BdNode_idx[map]
            K = bm.where(sBdNodeidx[:,None] == Verticesidx)[0]
            K = bm.concat([K,bm.array([len(sBdNodeidx)])])
            logic_bdnode = bm.zeros_like(node,**self.kwargs0)
            Pside_vector = bm.roll(physics_domain,-1,axis=0) - physics_domain
            Pside_length = bm.linalg.norm(Pside_vector,axis=1)
            rate = Lside_length / Pside_length
            theta = bm.arctan2(Lside_vector[:,1],Lside_vector[:,0]) -\
                    bm.arctan2(Pside_vector[:,1],Pside_vector[:,0])
            ctheta = bm.cos(theta)
            stheta = bm.sin(theta)
            R = bm.concat([ctheta,stheta,
                        -stheta,ctheta],axis=0).reshape(2,2,num_sides).T
            A = rate[:,None,None] * R
            A_repeat = bm.repeat(A,K[1:]-K[:-1],axis=0)
            PVertices_repeat = bm.repeat(physics_domain,K[1:]-K[:-1],axis=0)
            LVertices_repeat = bm.repeat(logic_domain,K[1:]-K[:-1],axis=0)
            Aim_vector = (A_repeat@((node[sBdNodeidx]-PVertices_repeat)[:,:,None])).reshape(-1,2)
            logic_bdnode = bm.set_at(logic_bdnode,sBdNodeidx,Aim_vector+LVertices_repeat)
            map = bm.where((node[:,None] == p).all(axis=2))[0]
            return logic_bdnode[map]
        
    def _get_logic_node_init(self) -> TensorLike:
        """
        get the initial guess of the logic node
        
        Returns
            TensorLike: initial logic node positions
        """
        if self.mesh_type in ["LagrangeTriangleMesh","LagrangeQuadrangleMesh"]:
            mesh = self.linermesh
        else:
            mesh = self.mesh
        bdc = self._get_logic_boundary
        # p = self.p 
        p = 1
        # space = self.space
        space = LagrangeFESpace(mesh, p=p)
        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=p+1,method=None))
        A = bform.assembly()
        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(source=0,q=p+1,method=None))
        F = lform.assembly()
        bc0 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,0])
        bc1 = DirichletBC(space = space, gd = lambda p : bdc(p)[:,1])
        uh0 = space.function()
        uh1 = space.function()
        A1, F1 = bc0.apply(A, F, uh0)
        A2, F2 = bc1.apply(A, F, uh1)
        uh0 = bm.set_at(uh0 , slice(None), spsolve(A1, F1 , solver=self.solver))
        uh1 = bm.set_at(uh1 , slice(None), spsolve(A2, F2 , solver=self.solver))
        logic_node = bm.concat([uh0[:,None],uh1[:,None]],axis=1)
        return logic_node
    
    def _get_method(self, method_name: str):
        """
        get method by name
        
        Parameters
            method_name: str, name of the method
        Returns
            method: corresponding method object
        """
        print(f"Getting method: {method_name}")
        if hasattr(self, method_name):
            return getattr(self, method_name)
        else:
            raise AttributeError(f"Method {method_name} not found")

 
class MM_monitor(MM_PREProcessor):
    def __init__(self,mesh,beta,space,config:Config) -> None:
        super().__init__(mesh,space,config)
        self.beta = beta
        self.r = config.r
        self.alpha = config.alpha
        self.mol_times = config.mol_times

        # self.monitor = config.monitor
        # self.mol_meth = config.mol_meth
    
    def _grad_uh(self):
        """
        pointwise gradient of the solution
        
        Returns
            TensorLike: gradient of solution
        """
        uh = self.uh
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        gphi = pspace.grad_basis(self.bcs) # change
        guh_incell = bm.einsum('cqid , ci -> cqd ',gphi,uh[pcell2dof])
        return guh_incell

    def _mp_grad_uh(self):
        """
        pointwise gradient of the solution for multiphysics
        
        Returns
            TensorLike: gradient of solution for multiphysics
        """
        uh = self.uh
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        gphi = pspace.grad_basis(self.bcs)
        guh_incell = bm.einsum('cqid , cil -> cqld ',gphi,uh[pcell2dof])
        return guh_incell

    @variantmethod('arc_length')
    def monitor(self):
        """
        arc length monitor
        """
        guh_incell = self._grad_uh()
        self.M = bm.sqrt(1 +  self.beta * bm.sum(guh_incell**2,axis=-1))
        
    @monitor.register('norm_arc_length')
    def monitor(self):
        """
        normalized arc length monitor
        """
        guh_incell = self._grad_uh()
        R = bm.max(bm.linalg.norm(guh_incell,axis=-1))
        if R <= 1e-15:
            R = 1
        self.M = bm.sqrt(1 + self.beta * bm.sum(guh_incell**2,axis=-1)/R**2)
    
    @monitor.register('mp_arc_length')
    def monitor(self):
        """
        arc length monitor for multiphysics
        """
        guh_incell = self._mp_grad_uh()
        self.M = bm.sqrt(1 + 1/self.dim*bm.sum(self.beta[None,None,:]*
                                   bm.sum(guh_incell**2,axis=-1),axis=-1))

    @monitor.register('matrix_normal')
    def monitor(self):
        """
        matrix normal monitor method
        """
        guh_incell = self._grad_uh() # NC,NQ,TD
        
        norm_guh_cell = bm.linalg.norm(guh_incell,axis=-1) # NC,NQ
        is_zero = norm_guh_cell < 1e-15
        v = bm.zeros_like(guh_incell,**self.kwargs0)
        v = bm.set_at(v, is_zero, bm.array([1.0,0.0],**self.kwargs0))
        v = bm.set_at(v, ~is_zero, guh_incell[~is_zero]/norm_guh_cell[~is_zero][...,None])
        v_orth = bm.stack([-v[..., 1], v[..., 0]], axis=-1)

        R = bm.sqrt(1 +  bm.sum(guh_incell**2,axis=-1))-1 # NC,NQ
        R_mean =(bm.einsum('q,cq,cq ->',self.ws,R, self.d*self.rm)/
                 bm.sum(self.cm))
        if bm.max(R_mean) < 1e-15:
            print("Warning: R_mean is too small, using default value.")
            R_mean = 1.0
        alpha = self.beta/(R_mean*(1.0-self.beta))
        lambda_1 = 1 + alpha*R # NC,NQ

        self.M = lambda_1[...,None,None]*v[...,None,:]*v[...,None] + \
                 lambda_1[...,None,None]*v_orth[...,None,:]*v_orth[...,None] # NC,NQ,TD,TD

    @variantmethod('projector')
    def mol_method(self):
        """
        projection operator mollification method
        """
        M = self.M
        exp_nd = M.ndim - 2
        cell2dof = self.cell2dof  # NC,NQ
        sm = self.sm
        d = self.d # NC,NQ
        rm = self.rm
        exp_sm = sm[(...,) + (None,) * exp_nd]
        shape = (self.NN,) + (self.TD,) * exp_nd
        phi = self.mspace.basis(self.bcs)
        dphi = phi*rm*d[(...,) + (None,) *(3-d.ndim)]  # NC,NQ,...
        M = M*rm*d[(...,) + (None,) * (2-d.ndim+exp_nd)]  # NC,NQ,...
        M_node = bm.zeros(shape, **self.kwargs0)
        for i in range(self.mol_times):
            if i != 0:
                M = bm.einsum('cqi,ci...->cq...', dphi, M_node[cell2dof])
            M_incell = bm.einsum('cq...,q-> c...',M,self.ws)
            M_node.fill(0)
            M_node = bm.index_add(M_node, cell2dof, M_incell[:,None,...])
            M_node /= exp_sm
            
        self.M = bm.einsum('cqi,ci...->cq...', phi, M_node[cell2dof])
        self.M_node = M_node
    
    @mol_method.register('heat_equ')
    def mol_method(self):
        """
        heat equation mollification method
        """
        M = self.M
        h = self.hmin
        r = self.r
        R = r*(1+r)
        dt = 1/self.mol_times
        mass = self.mass
        pspace = self.pspace
        bform = BilinearForm(pspace)
        lform = LinearForm(pspace)
        SDI = self.SDI
        SSI = self.SSI
        
        bform.add_integrator(SDI)
        lform.add_integrator(SSI)
        SDI.coef = h**2*R*dt
        SDI.clear()
        M_bar = pspace.function()
        A = bform.assembly() + mass
        for i in range(self.mol_times):
            SSI.source = M
            SSI.clear()
            b = lform.assembly()
            M_bar[:] = cg(A,b,atol=1e-5,returninfo=True)[0]
            M = M_bar(self.bcs)
        self.M = M


class MM_Interpolater(MM_PREProcessor):
    def __init__(self, mesh,space, config: Config):
        super().__init__(mesh,space, config)
        self.pde = config.pde
        self.int_meth = config.int_meth  # by setter initialize _int_meth

        if self.mesh_type == "TriangleMesh":
            self.interpolate_batch = self._tri_interpolate_batch
            self.high_order_batch = self._tri_high_order_interpolate
        elif self.mesh_type == "QuadrangleMesh":
            self.interpolate_batch = self._quad_interpolate_batch
            self.high_order_batch = self._quad_high_order_interpolate
        elif self.mesh_type == "TetrahedronMesh":
            self.interpolate_batch = self._tet_interpolate_batch
            self.high_order_batch = self._tet_high_order_interpolate
        elif self.mesh_type == "HexahedronMesh":
            self.interpolate_batch = self._hex_interpolate_batch
            self.high_order_batch = self._hex_high_order_interpolate

    @variantmethod('comass')
    def interpolate(self,moved_node:TensorLike):
        """
        conservation of mass interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        Returns
            TensorLike: interpolated solution
        """
        delta_x = self.node - moved_node
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        if pspace.p > self.p:
            delta_x = self.high_order_batch(delta_x, pspace.p)  # 高阶插值处理

        bcs = self.bcs
        ws = self.ws
        phi = pspace.basis(bcs)
        gphi = pspace.grad_basis(bcs)
        GDOF = pspace.number_of_global_dofs()
        rm = self.rm
        cm = self.d * rm

        M = self.mass
        P = bm.einsum('...,c...id,cid,c...j ,c... -> cij',ws, gphi,delta_x[pcell2dof],phi,cm)
        
        I,J = self.I,self.J
        indices = bm.stack([I.ravel(), J.ravel()], axis=0)
        P = COOTensor(indices=indices, values=P.ravel(), spshape=(GDOF,GDOF))
        P.tocsr()

        def ODEs(t,y):
            y = bm.asarray(y,**self.kwargs0)
            # f = spsolve(M, P @ y, solver=self.solver)
            f = cg(M, P @ y, atol=1e-8,returninfo=True)[0]
            return f
        
        sol = solve_ivp(ODEs,[0,1],y0=self.uh,method='RK23').y[:,-1]
        sol = bm.asarray(sol,**self.kwargs0)
        return sol

    @interpolate.register('linear')
    def interpolate(self, moved_node: TensorLike):
        """
        linear interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        """
        node2cell = self.node2cell
        i, j = node2cell.row, node2cell.col
        p = self.pspace.p # physical space polynomial degree
        new_uh = bm.zeros(self.NN, **self.kwargs0) # 初始化新的解向量,对节点优先进行赋值
        interpolated = bm.zeros(self.NN, dtype=bool, device=self.device)
        
        current_i, current_j = self.interpolate_batch(i, j, new_uh, 
                                                      interpolated,moved_node) 
        # 迭代扩展 - 添加循环上限
        max_iterations = min(30, int(bm.log(self.NC)) + 20)
        iteration_count = 0
        # 迭代扩展
        while len(current_i) > 0 and iteration_count < max_iterations:
            iteration_count += 1
            # 扩展邻居
            neighbors = self.cell2cell[current_j].flatten()
            expanded_i = bm.repeat(current_i, self.cell2cell.shape[1])
            valid_mask = neighbors >= 0

            if not bm.any(valid_mask):
                break

            combined = expanded_i[valid_mask] * self.NC + neighbors[valid_mask]
            unique_combined = bm.unique(combined)
            
            unique_i = unique_combined // self.NC
            unique_j = unique_combined % self.NC
            current_i, current_j = self.interpolate_batch(unique_i, unique_j,
                                                    new_uh, interpolated,moved_node)
        if iteration_count >= max_iterations:
            print(f"Warning: Maximum iterations reached ({max_iterations}) without full interpolation.")
        
        new_uh = self.high_order_batch(new_uh, p)  # 高阶插值处理
        return new_uh
        
    def _tri_interpolate_batch(self,nodes, cells,new_uh, interpolated,moved_node):
        """
        triangle mesh interpolation batch processing
        
        Parameters
            nodes: TensorLike, nodes to be interpolated
            cells: TensorLike, cells corresponding to the nodes
            new_uh: TensorLike, new solution vector
            interpolated: TensorLike, boolean mask indicating if nodes are already interpolated
            moved_node: TensorLike, moved node positions
        Returns
            nodes: TensorLike, nodes that still need interpolation
            cells: TensorLike, cells corresponding to the nodes that still need interpolation
        """
        if len(nodes) == 0:
            return bm.array([], **self.kwargs1), bm.array([], **self.kwargs1)
            
        # 计算重心坐标
        v_matrix = bm.permute_dims(
            self.node[self.cell[cells, 1:]] - self.node[self.cell[cells, 0:1]], 
            axes=(0, 2, 1)
        )
        v_b = moved_node[nodes] - self.node[self.cell[cells, 0]]
        
        inv_matrix = bm.linalg.inv(v_matrix)
        lam = bm.einsum('cij,cj->ci', inv_matrix, v_b)
        lam = bm.concat([(1 - bm.sum(lam, axis=-1, keepdims=True)), lam], axis=-1)
        valid = bm.all(lam > -1e-10, axis=-1) & ~interpolated[nodes]
        
        if bm.any(valid):
            valid_nodes = nodes[valid]
            phi = self.mesh.shape_function(lam[valid], self.pspace.p)
            valid_value = bm.sum(phi * self.uh[self.pcell2dof[cells[valid]]], axis=1)
            
            new_uh = bm.set_at(new_uh, valid_nodes, valid_value)
            interpolated = bm.set_at(interpolated, valid_nodes, True)
        
        return nodes[~interpolated[nodes]], cells[~interpolated[nodes]]
            
    def _quad_interpolate_batch(self,nodes, cells,new_uh, interpolated,moved_node):
        """
        Quadrilateral mesh interpolation batch processing
        
        Parameters
            nodes: TensorLike, nodes to be interpolated
            cells: TensorLike, cells corresponding to the nodes
            new_uh: TensorLike, new solution vector
            interpolated: TensorLike, boolean mask indicating if nodes are already interpolated
            moved_node: TensorLike, moved node positions
        Returns
            nodes: TensorLike, nodes that still need interpolation
            cells: TensorLike, cells corresponding to the nodes that still need interpolation
        """
        if len(nodes) == 0:
            return bm.array([], **self.kwargs1), bm.array([], **self.kwargs1)
        
        # 使用 [0,1]×[0,1] 参数坐标求解
        xi_eta = _solve_quad_parametric_coords(
            moved_node[nodes], self.node[self.cell[cells]]
        )
        # 检查有效性：参数在 [0,1]×[0,1] 范围内
        tolerance = 1e-6
        param_in_range = bm.all((xi_eta >= -tolerance) & (xi_eta <= 1.0 + tolerance), axis=-1)
        not_interpolated = ~interpolated[nodes]

        valid = (param_in_range & not_interpolated)
        if bm.any(valid):
            mi = bm.multi_index_matrix(self.pspace.p, 1, dtype=self.itype)
            bc = (bm.stack([1-xi_eta[valid,0],xi_eta[valid,0]], axis=1),
                  bm.stack([1-xi_eta[valid,1],xi_eta[valid,1]], axis=1))
            phi0 = bm.simplex_shape_function(bc[0],p=self.pspace.p,mi=mi)
            phi1 = bm.simplex_shape_function(bc[1],p=self.pspace.p,mi=mi)
            
            valid_nodes = nodes[valid]
            valid_cells = cells[valid]
            valid_shape = (phi0[:, :, None] * phi1[:, None, :]).reshape(phi0.shape[0], -1)
            valid_value = bm.sum(valid_shape * self.uh[self.pcell2dof[valid_cells]], axis=1)
            new_uh = bm.set_at(new_uh, valid_nodes, valid_value)
            interpolated = bm.set_at(interpolated, valid_nodes, True)

        return nodes[~interpolated[nodes]], cells[~interpolated[nodes]]
    
    def _tet_interpolate_batch(self, nodes, cells, new_uh, interpolated, moved_node):
        """
        Tetrahedral mesh interpolation batch processing
        
        Parameters
            nodes: TensorLike, nodes to be interpolated
            cells: TensorLike, cells corresponding to the nodes
            new_uh: TensorLike, new solution vector
            interpolated: TensorLike, boolean mask indicating if nodes are already interpolated
            moved_node: TensorLike, moved node positions
        Returns
            nodes: TensorLike, nodes that still need interpolation
            cells: TensorLike, cells corresponding to the nodes that still need interpolation
        """
        if len(nodes) == 0:
            return bm.array([], **self.kwargs1), bm.array([], **self.kwargs1)
        # 计算重心坐标 (类似三角形但在3D)
        # 对四面体 [v0, v1, v2, v3]，重心坐标通过求解 3x3 线性系统
        v_matrix = bm.permute_dims(
            self.node[self.cell[cells, 1:]] - self.node[self.cell[cells, 0:1]], 
            axes=(0, 2, 1)
        )  # (NC, 3, 3)
        v_b = moved_node[nodes] - self.node[self.cell[cells, 0]]  # (NC, 3)
        
        inv_matrix = bm.linalg.inv(v_matrix)
        lam = bm.einsum('cij,cj->ci', inv_matrix, v_b)  # (NC, 3)
        lam = bm.concat([(1 - bm.sum(lam, axis=-1, keepdims=True)), lam], axis=-1)  # (NC, 4)
        
        # 检查重心坐标有效性
        valid = bm.all(lam > -2e-5, axis=-1) & ~interpolated[nodes]
        if bm.any(valid):
            valid_nodes = nodes[valid]
            # 使用四面体形函数
            phi = self.mesh.shape_function(lam[valid], self.pspace.p)
            valid_value = bm.sum(phi * self.uh[self.pcell2dof[cells[valid]]], axis=1)
            
            new_uh = bm.set_at(new_uh, valid_nodes, valid_value)
            interpolated = bm.set_at(interpolated, valid_nodes, True)
        
        return nodes[~interpolated[nodes]], cells[~interpolated[nodes]]
    
    def _hex_interpolate_batch(self, nodes, cells, new_uh, interpolated, moved_node):
        """
        Hexahedral mesh interpolation batch processing
        
        Parameters
            nodes: TensorLike, nodes to be interpolated
            cells: TensorLike, cells corresponding to the nodes
            new_uh: TensorLike, new solution vector
            interpolated: TensorLike, boolean mask indicating if nodes are already interpolated
            moved_node: TensorLike, moved node positions
        Returns
            nodes: TensorLike, nodes that still need interpolation
            cells: TensorLike, cells corresponding to the nodes that still need interpolation
        """
        if len(nodes) == 0:
            return bm.array([], **self.kwargs1), bm.array([], **self.kwargs1)
            
        # 使用 [0,1]³ 参数坐标求解六面体内的位置
        xi_eta_zeta = _solve_hex_parametric_coords(
            moved_node[nodes], self.node[self.cell[cells]]
        )
        # 检查有效性：参数在 [0,1]³ 范围内
        tolerance = 5e-6
        param_in_range = bm.all((xi_eta_zeta >= -tolerance) & (xi_eta_zeta <= 1.0 + tolerance), axis=-1)
        not_interpolated = ~interpolated[nodes]

        valid = (param_in_range & not_interpolated)

        if bm.any(valid):
            # 构造三线性形函数
            xi, eta, zeta = xi_eta_zeta[valid, 0], xi_eta_zeta[valid, 1], xi_eta_zeta[valid, 2]
            # 对于高阶六面体，需要使用张量积形函数
            mi = bm.multi_index_matrix(self.pspace.p, 1, dtype=self.itype)
            bc = (bm.stack([1-xi, xi], axis=1),
                  bm.stack([1-eta, eta], axis=1),
                  bm.stack([1-zeta, zeta], axis=1))
            phi0 = bm.simplex_shape_function(bc[0], p=self.pspace.p, mi=mi)
            phi1 = bm.simplex_shape_function(bc[1], p=self.pspace.p, mi=mi)
            phi2 = bm.simplex_shape_function(bc[2], p=self.pspace.p, mi=mi)
            # 张量积形函数
            valid_shape = (phi0[:, :, None, None] * 
                           phi1[:, None, :, None] * 
                           phi2[:, None, None, :]).reshape(phi0.shape[0], -1)
            valid_nodes = nodes[valid]
            valid_cells = cells[valid]
            valid_value = bm.sum(valid_shape * self.uh[self.pcell2dof[valid_cells]], axis=1)
            new_uh = bm.set_at(new_uh, valid_nodes, valid_value)
            interpolated = bm.set_at(interpolated, valid_nodes, True)

        return nodes[~interpolated[nodes]], cells[~interpolated[nodes]]
    
    def _tri_high_order_interpolate(self,value,p):
        """
        high order interpolation for triangle mesh(p >= 2)

        Parameters
            new_uh: Tensor, the new solution vector
            p: int, polynomial degree of the space
        Returns
            TensorLike: new_uh with high order interpolation values added
        """
        if p >= 2:
            edge = self.mesh.edge
            w = self.mesh.multi_index_matrix(p, 1, dtype=self.ftype,device=self.device)
            w = w[1:-1]/p
            edge_value = bm.einsum('ij,cj...->ci...', w,value[edge]).reshape(-1,*value.shape[1:])
            value = bm.concat((value, edge_value), axis=0)

        if p >= 3:
            TD = self.TD
            cell = self.cell
            multiIndex = self.mesh.multi_index_matrix(p, TD, dtype=self.ftype,device=self.device)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex / p
            cell_value = bm.einsum('ij, kj...->ki...', w,value[cell]).reshape(-1,*value.shape[1:])
            value = bm.concat((value, cell_value), axis=0)
        
        return value

    def _quad_high_order_interpolate(self, value,p):
        """
        high order interpolation for quadrilateral mesh(p >= 2)

        Parameters
            new_uh: Tensor, the new solution vector
            p: int, polynomial degree of the space
        Returns
            TensorLike: new_uh with high order interpolation values added
        """
        if p >= 2:
            cell = self.cell
            edge = self.mesh.edge
            multiIndex = self.mesh.multi_index_matrix(p, 1, dtype=self.ftype, device=self.device)
            w = multiIndex[1:-1, :] / p
            edge_value = bm.einsum('ij,cj...->ci...', w, value[edge]).reshape(-1,*value.shape[1:])
            w = bm.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
            cell_value = bm.einsum('ij, kj...->ki...', w, value[cell[:,[0,3,1,2]]]).reshape(-1,*value.shape[1:])
            value = bm.concat((value, edge_value, cell_value), axis=0)
        
        return value

    def _tet_high_order_interpolate(self, value, p):
        """
        High order interpolation for tetrahedral mesh (p >= 2)

        Parameters
            value: Tensor, the solution vector
            p: int, polynomial degree of the space
        Returns
            TensorLike: value with high order interpolation values added
        """
        if p >= 2:
            # 边上的插值点
            edge = self.mesh.edge
            w = self.mesh.multi_index_matrix(p, 1, dtype=self.ftype, device=self.device)
            w = w[1:-1]/p
            edge_value = bm.einsum('ij,cj...->ci...', w, value[edge]).reshape(-1, *value.shape[1:])
            value = bm.concat((value, edge_value), axis=0)

        if p >= 3:
            # 面上的插值点
            face = self.mesh.face
            multiIndex_face = self.mesh.multi_index_matrix(p, 2, dtype=self.ftype, device=self.device)
            isEdgeIPoints_face = (multiIndex_face == 0)
            isInFaceIPoints = ~(isEdgeIPoints_face[:, 0] | isEdgeIPoints_face[:, 1] | 
                               isEdgeIPoints_face[:, 2])
            multiIndex_face = multiIndex_face[isInFaceIPoints, :]
            w_face = multiIndex_face / p
            face_value = bm.einsum('ij, kj...->ki...', w_face, value[face]).reshape(-1, *value.shape[1:])
            value = bm.concat((value, face_value), axis=0)

        if p >= 4:
            # 体内的插值点
            TD = self.TD  # 应该是3
            cell = self.cell
            multiIndex = self.mesh.multi_index_matrix(p, TD, dtype=self.ftype, device=self.device)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2] | isEdgeIPoints[:, 3])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex / p
            cell_value = bm.einsum('ij, kj...->ki...', w, value[cell]).reshape(-1, *value.shape[1:])
            value = bm.concat((value, cell_value), axis=0)
        
        return value
    
    def _hex_high_order_interpolate(self, value, p):
        """
        High order interpolation for hexahedral mesh (p >= 2)

        Parameters
            value: Tensor, the solution vector
            p: int, polynomial degree of the space
        Returns
            TensorLike: value with high order interpolation values added
        """
        if p >= 2:
            cell = self.cell
            edge = self.mesh.edge
            multiIndex = self.mesh.multi_index_matrix(p, 1, dtype=self.ftype, device=self.device)
            w = multiIndex[1:-1, :] / p
            edge_value = bm.einsum('ij,cj...->ci...', w, value[edge]).reshape(-1, *value.shape[1:])
            
            # 面上的高阶点 (每个面是四边形)
            face = self.mesh.face
            w_face = bm.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
            # 重新排列面节点顺序以匹配参数坐标
            face_value = bm.einsum('ij, kj...->ki...', w_face, 
                                  value[face[:, [0, 1, 2, 3]]]).reshape(-1, *value.shape[1:])
            # 体内的高阶点 (三重张量积)
            w_cell = bm.einsum('im, jn, ko->ijkimno', w, w, w).reshape(-1, 8)
            # 重新排列单元节点顺序
            cell_value = bm.einsum('ij, kj...->ki...', w_cell, 
                                  value[cell[:, [0, 1, 2, 3, 4, 5, 6, 7]]]).reshape(-1, *value.shape[1:])
            
            value = bm.concat((value, edge_value, face_value, cell_value), axis=0)
        
        return value
    
    @interpolate.register('mp_comass')
    def interpolate(self,moved_node:TensorLike):
        """
        conservation of mass interpolation method for multiphysics
        
        Parameters
            moved_node: TensorLike, new node positions
        Returns
            TensorLike: interpolated solution for multiphysics
        """
        delta_x = self.node - moved_node
        pspace = self.pspace
        pcell2dof = self.pcell2dof
        
        bcs = self.bcs
        ws = self.ws
        phi = pspace.basis(bcs)
        gphi = pspace.grad_basis(bcs)
        GDOF = pspace.number_of_global_dofs()
        rm = self.rm
        cm = self.d * rm

        M = self.mass
        P = bm.einsum('...,c...id,cid,c...j ,c... -> cij',ws, gphi,delta_x[pcell2dof],phi,cm)
        
        I,J = self.I,self.J
        indices = bm.stack([I.ravel(), J.ravel()], axis=0)
        P = COOTensor(indices=indices, values=P.ravel(), spshape=(GDOF,GDOF))
        P.tocsr()

        def ODEs(t,y):
            y = bm.asarray(y,**self.kwargs0)
            f = cg(M, P @ y, atol=1e-8,returninfo=True)[0]
            return f
        
        sol = bm.zeros_like(self.uh,**self.kwargs0)
        for i in range(self.dim):
            s = solve_ivp(ODEs,[0,1],y0=self.uh[:,i],method='RK23').y[:,-1]
            sol = bm.set_at(sol,(...,i),s)
        return sol
    
    @interpolate.register('solution')
    def interpolate(self,moved_node:TensorLike):
        """
        get the solution
        
        Parameters
            moved_node: TensorLike, new node positions
        Returns
            TensorLike: solution values
        """
        pde = self.pde
        return pde.init_solution(moved_node)
    
    @interpolate.register('poisson')
    def interpolate(self,moved_node:TensorLike):
        """
        poisson interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        Returns
            TensorLike: solution from Poisson equation
        """
        pde = self.pde
        pspace = self.pspace
        self.mesh.node = moved_node
        bform = BilinearForm(self.pspace)
        lform = LinearForm(self.pspace)
        SDI = self.SDI
        SSI = self.SSI
        SDI.coef = 1
        SDI.clear()
        SSI.source = pde.source
        SSI.clear()
        bform.add_integrator(SDI)
        lform.add_integrator(SSI)
        A = bform.assembly()
        b = lform.assembly()
        bc = DirichletBC(self.pspace, gd=pde.dirichlet)
        A, b = bc.apply(A, b)
        return spsolve(A, b, solver=self.solver)
    
    @interpolate.register('convect_diff')
    def interpolate(self,moved_node:TensorLike):
        """
        convection diffusion interpolation method
        
        Parameters
            moved_node: TensorLike, new node positions
        Returns
            TensorLike: solution from convection-diffusion equation
        """
        pde = self.pde
        mesh = self.mesh
        mesh.node = moved_node
        am = self.assambly_method
        self.pspace.mesh = mesh
        source = pde.source(mesh.node)
        bcs = self.bcs
        source = self.pspace.value(source, bcs)
        q = self.p+2
        a = pde.a[None,None,:]
        b = pde.b
        SDI = self.SDI
        SSI = self.SSI
        SCI = ScalarConvectionIntegrator(coef = a, q = q, method=am)
        SDI.coef = b
        SDI.clear()
        SSI.source = source
        SSI.clear()
        bform = BilinearForm(self.pspace)
        lform = LinearForm(self.pspace)
        bform.add_integrator(SDI,SCI)
        lform.add_integrator(SSI)
        A = bform.assembly()
        b = lform.assembly()
        bc = DirichletBC(self.pspace, pde.dirichlet)
        A,b = bc.apply(A,b)
        return spsolve(A, b,  solver=self.solver)
