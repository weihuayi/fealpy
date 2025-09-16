from ..decorator import variantmethod
from . import PREProcessor
from .config import *
from .tool import (_solve_quad_parametric_coords,
                   _solve_hex_parametric_coords,
                   newton_barycentric_triangle)


class Interpolater(PREProcessor):
    def __init__(self,mesh,space,config: Config, **kwargs):
        super().__init__(mesh = mesh, space = space, config = config, **kwargs)
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
    
    # @variantmethod('comass')
    # def interpolate(self,moved_node:TensorLike):
    #     """
    #     conservation of mass interpolation method
        
    #     Parameters
    #         moved_node: TensorLike, new node positions
    #     Returns
    #         TensorLike: interpolated solution
    #     """
    #     delta_x = self.node - moved_node
    #     pspace = self.pspace
    #     pcell2dof = self.pcell2dof
    #     if pspace.p > self.p:
    #         delta_x = self.high_order_batch(delta_x, pspace.p)  # 高阶插值处理

    #     bcs = self.bcs
    #     ws = self.ws
    #     phi = pspace.basis(bcs)
    #     gphi = pspace.grad_basis(bcs)
    #     GDOF = pspace.number_of_global_dofs()
    #     rm = self.rm
    #     cm = self.d * rm
    #     M = self.mass
        
    #     d_v = bm.einsum('cqi , cid -> cqd', phi, delta_x[pcell2dof])
    #     # P = bm.einsum('...,c...id,cid,c...j ,c... -> cij',ws, gphi,delta_x[pcell2dof],phi,cm)

    #     # I,J = self.I,self.J
    #     # indices = bm.stack([I.ravel(), J.ravel()], axis=0)
    #     # P = COOTensor(indices=indices, values=P.ravel(), spshape=(GDOF,GDOF))
    #     # P.tocsr()
        
    #     def ODEs(t,y):
             
    #         y = bm.asarray(y,**self.kwargs0)
    #         source = bm.einsum('cqid , ci ,cqd -> cq ',gphi,y[pcell2dof],d_v)
    #         f = bm.einsum('q,cq,cqi,cq->ci', ws, source,phi, cm)
    #         k = bm.zeros_like(y, **self.kwargs0)
    #         k = bm.index_add(k, pcell2dof, f)
    #         # f = spsolve(M, P @ y, solver=self.solver)
    #         f = cg(M, k, atol=1e-8,returninfo=True)[0]
    #         return f
        
    #     sol = solve_ivp(ODEs,[0,1],y0=self.uh,method='RK23').y[:,-1]
    #     sol = bm.asarray(sol,**self.kwargs0)
    #     return sol

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
        tolerance = 2e-5
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
    
    @interpolate.register('lagrangetri_2d')
    def interpolate(self,moved_node:TensorLike,
                    local_vnode_idx, local_enode_idx , local_cnode_idx,
                    global_vnode_idx, global_enode_idx , global_cnode_idx):
        """
        """
        p_field = self.pspace.p                # 解空间阶次
        p_geom = self.lmspace.p                # 几何映射阶次
        new_uh = bm.zeros(self.NN, **self.kwargs0)
        interpolated = bm.zeros(self.NN, dtype=bm.bool, device=self.device)

        shape_func_geom = self.mesh.shape_function         # 几何映射形函数
        grad_shape_func_geom = self.mesh.grad_shape_function

        # 批量牛顿 + 赋值 节点-多单元候选
        def batch_locate_and_assign(part_idx, local_vnode_idx, node2cell_sub, interpolated, new_uh,mark):
            """
            global_nodes: 需要插值的一类节点(顶点/边/内部)的全局索引 (Gn,)
            node2cell_sub: 该类节点的 node2cell 切片 (Gn, max_valence) 的 CSR-like 提取 (使用 row,col)
            local_index_selector: 提供单元局部自由度全索引 (和 self.cell[idx] 对应)
            """
            gi, gc = node2cell_sub.row, node2cell_sub.col      # 每个 (节点, 候选单元) 对
            gi = part_idx[gi]                                # 全局节点索引
            if len(gi) == 0:
                return new_uh, interpolated
            # 候选物理点；几何单元坐标
            tgt = moved_node[gi]                          # (M,2)
            cell_nodes = self.node[self.cell[gc]]         # (M, Ndof_geom, 2)
            # 调用牛顿
            lam, success = newton_barycentric_triangle(
                target=tgt,
                cell_nodes=cell_nodes,
                p=p_geom,
                local_vnode_idx=local_vnode_idx,
                shape_function=shape_func_geom,
                grad_shape_function=grad_shape_func_geom,
                tol=1e-6,
                maxit=100,
                damping=True,
            )
            # 有效性：收敛且 lam>= -tol
            tol_in = 1e-6
            lam_valid = success & bm.all(lam >= -tol_in, axis=1)
            if not bm.any(lam_valid):
                print(f"[lagrangetri_2d {mark}] Warning: No valid interpolation points found in lam_valid.")
                return new_uh, interpolated

            # 过滤掉已插值节点
            need = ~interpolated[gi]
            eff = lam_valid & need
            if not bm.any(eff):
                print(f"[lagrangetri_2d {mark}] Warning: All valid points have been interpolated already in this batch.")
                return new_uh, interpolated

            # 这里使用“出现顺序第一条”策略
            eff_idx = bm.where(eff)[0]
            gi_eff = gi[eff_idx]
            _u, first_pos = bm.unique(gi_eff, return_index=True)  
            pick_idx = eff_idx[first_pos]
           
            if len(pick_idx) == 0:
                print(f"[lagrangetri_2d {mark}] Warning: No unique nodes to interpolate in pick_idx.")
                return new_uh, interpolated

            lam_pick = lam[pick_idx]                 # (S,3)
            cell_pick = gc[pick_idx]                # (S,)
            node_pick_global = gi[pick_idx]         # (S,)

            # 计算场形函数
            phi = shape_func_geom(lam_pick, p_field)[0]  # (S, ldof_field)
            dofs = self.pcell2dof[cell_pick]                   # (S, ldof_field)

            uh_local = self.uh[dofs]                           # (S, ldof_field) 或 (S, ldof_field, ncomp)
            # 兼容多分量
            if uh_local.ndim == 2:
                vals = bm.sum(phi * uh_local, axis=1)# (S,)
                new_uh = bm.set_at(new_uh, node_pick_global, vals)
            else:
                # uh_local: (S, ldof_field, ncomp)
                vals = bm.einsum('...si,sic->sc', phi, uh_local)  # (S,ncomp)
                # 若 new_uh 需要多分量，需在外面初始化为 (NN, ncomp)
                new_uh = bm.set_at(new_uh, node_pick_global[:, None], vals)

            interpolated = bm.set_at(interpolated, node_pick_global, True)
            return new_uh, interpolated
        
        node2cell = self.node2cell
        
        # 1) 顶点
        vnode2cell = node2cell[global_vnode_idx, :]
        new_uh, interpolated = batch_locate_and_assign(global_vnode_idx,
        local_vnode_idx,vnode2cell,interpolated,new_uh,"v")
        # 2) 边节点
        enode_mask = ~interpolated[global_enode_idx]
        if bm.any(enode_mask):
            enode2cell = node2cell[global_enode_idx[enode_mask], :]
            new_uh, interpolated = batch_locate_and_assign(global_enode_idx[enode_mask], 
            local_vnode_idx, enode2cell, interpolated, new_uh, "e")
        # 3) 内部节点
        cnode_mask = ~interpolated[global_cnode_idx]
        if bm.any(cnode_mask):
            cnode2cell = node2cell[global_cnode_idx[cnode_mask], :]
            new_uh, interpolated = batch_locate_and_assign(global_cnode_idx[cnode_mask],
            local_vnode_idx,cnode2cell,interpolated,new_uh,"c")

        remaining = bm.where(~interpolated)[0]
        if len(remaining) > 0:
            fb = node2cell[remaining, :]
            new_uh, interpolated = batch_locate_and_assign(remaining, local_vnode_idx, fb, interpolated, new_uh,"r")
            still = bm.where(~interpolated)[0]
            if len(still) > 0:
                print(f"[lagrangetri_2d] Warning: {len(still)} nodes not interpolated!")

        # TODO: 高阶插值需要额外处理
        return new_uh

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
        space = self.pspace
        uh = space.interpolate(pde.init_solution)
        return uh

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
        am = self.assembly_method
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