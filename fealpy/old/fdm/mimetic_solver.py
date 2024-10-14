import numpy as np

from fealpy.decorator import cartesian
from scipy.sparse import diags, lil_matrix,csr_matrix
from scipy.sparse import spdiags

class Mimetic():
    def __init__(self, mesh):
        self.mesh = mesh
    
    @cartesian
    def fun(self, p, index=None):
        x = p[..., 0]
        y = p[..., 1]
        val = x + y
        #val = (1+x) * (1+y)
        #val = np.sin(np.pi*x)*np.sin(np.pi*y)

        return val

    def gmv(self):
        """
        MV 质量矩阵

        Results
        - result (ndarray, (NN, NN) )
        """
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge_centers = mesh.entity_barycenter(etype=1)
        edge_unit_normals = mesh.edge_unit_normal()
        cell2node = mesh.ds.cell_to_node()
        cell2edge = mesh.ds.cell_to_edge()
        flag = np.where(mesh.ds.cell_to_edge_sign().toarray(), 1, -1)

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        edge_measure = mesh.entity_measure(etype=1)
        cell_measure = mesh.entity_measure(etype=2)
        edge_centers = mesh.entity_barycenter(etype=1)
        cell_centers = mesh.entity_barycenter(etype=2) # (NC, GD)
        MV = np.zeros((NN, NN))

        LHS = []
        error = []
        for i in range(NC):
            LNE = len(cell2edge[i])
            cell_edge_unit_normals = edge_unit_normals[cell2edge[i]] # (LNE, GD)
            #print("cell_edge_unit_normals:\n", cell_edge_unit_normals)
            tmp = flag[i, cell2edge[i]].reshape(-1, 1) # (LNE, 1)
            # 单位外法向量
            cell_unit_outward_normals = cell_edge_unit_normals * tmp # (LNE, GD)
            #print("cell_unit_outward_normals:", cell_unit_outward_normals.shape, "\n", cell_unit_outward_normals)
            cell_edge_measure = edge_measure[cell2edge[i]].reshape(-1, 1) # (LNE, 1)
            #print("cell_edge_measure:", cell_edge_measure)
            cell_edge_centers = edge_centers[cell2edge[i]] # (LNE, GD)

            val = self.fun(node[cell2node[i]])
            # 边上两个顶点求和
            edge_values = []
            for edge_index in cell2edge[i]:
                edge_nodes = edge[edge_index]
                edge_value = val[cell2node[i].tolist().index(edge_nodes[0])] + val[cell2node[i].tolist().index(edge_nodes[1])]
                edge_values.append(edge_value)
            #print("edge_value:", np.array(edge_values))

            R = 0.25 * np.einsum('lg, lg, lk -> lk', cell_unit_outward_normals, \
                            cell_edge_centers - cell_centers[i, :], cell_edge_measure) # (LNE, 1)
            #print("R:", R.shape, "\n", R)
            LHS.append(np.einsum('lk, l ->', R, np.array(edge_values)))

            N = np.ones(len(cell2edge[i])).reshape(-1, 1) # (LNE, 1)
            #print("N:", N.shape, "\n", N)

            M_consistency = R @ np.linalg.inv(R.T @ N) @ R.T
            #M_stability = np.trace(R @ R.T) / cell_measure[i] * \
            #                                  (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M_stability = np.trace(R @ R.T) / cell_measure[i] / LNE * \
                                              (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M = M_consistency + M_stability # (LNE, LNE)
            #print("M:", M.shape, "\n", M.round(3))

            err = np.max(np.abs(M@N - R))
            error.append(err)
            indexi, indexj = np.meshgrid(cell2node[i], cell2node[i])
            MV[indexi, indexj] += M

        #return MV, np.array(LHS), np.array(error)
        return MV

    def gme(self):
        """
        ME 质量矩阵
        
        Results
        - result (ndarray, (NE, NE) )
        """
        mesh = self.mesh

        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        node = mesh.entity('node')
        cell2node = mesh.ds.cell_to_node()
        #print("cell2node:", cell2node[4])
        cell2edge = mesh.ds.cell_to_edge()
        #print("cell2edge:", cell2edge[4])
        edge2node = mesh.ds.edge_to_node()
        #print("edge2node:", edge2node)

        edge_measure = mesh.entity_measure(etype=1)
        #print("edge_measure:", edge_measure.shape, "\n", edge_measure)
        cell_measure = mesh.entity_measure(etype=2)

        edge_tagnet = node[edge2node[:, 1]] - node[edge2node[:, 0]] # (NE, GD)
        edge_unit_tagnet = edge_tagnet / edge_measure[:, np.newaxis]
        #print("edge_unit_tagnet:", edge_unit_tagnet.shape, "\n", edge_unit_tagnet) # (NE, GD)

        edge_centers = mesh.entity_barycenter(etype=1)
        cell_centers = mesh.entity_barycenter(etype=2) # (NC, GD)


        ME = np.zeros((NE, NE))

        grad_h = self.grad_operator()
        #print("grad_h:", grad_h.shape, "\n", grad_h.round(3))
        LHS = []
        error = []
        for i in range(NC):
        #for i in range(4,5):
            # 根据单元内边的全局编号提取行
            local_grad_h_rows = grad_h[cell2edge[i]]
            
            # 提取单元内节点的全局编号，并构建映射表
            global_to_local = {g: l for l, g in enumerate(cell2node[i])}

            # 构造单元内的梯度算子
            local_grad_h = np.zeros((len(cell2edge[i]), len(cell2node[i])))
            for j, row in enumerate(local_grad_h_rows):
                for global_idx, value in enumerate(row):
                    if global_idx in global_to_local:
                        local_grad_h[j, global_to_local[global_idx]] = value

            #print("local_grad_h:\n", local_grad_h.shape, "\n", local_grad_h.round(3))

            p_ch = self.fun(node[cell2node[i]])
            #print("p_ch:", p_ch.shape, "\n", p_ch.round(3))

            LNE = len(cell2edge[i])
            cell_edge_measure = edge_measure[cell2edge[i]] # (LNE, )
            cell_edge_centers = edge_centers[cell2edge[i]] # (LNE, GD)

            distVec = cell_edge_centers - cell_centers[i, :] # (LNE, GD)
            #print("distVec:", distVec.shape, "\n", distVec)
            rot_distVec = np.hstack(( [distVec[:, 1][:, np.newaxis], \
                                      -distVec[:, 0][:, np.newaxis]] )) # (LNE, GD)
            #print("rot_distVec:", rot_distVec.shape, "\n", rot_distVec)
            N = edge_unit_tagnet[cell2edge[i], :] # (LNE, GD)
            #print("N:", N.shape, "\n", N.round(3))

            cell2node0 = np.append(cell2node[i], cell2node[i][0])
            cell_edge_unit_tagnet = (node[cell2node0[1:]] - node[cell2node0[0:-1]])\
                / cell_edge_measure[:, np.newaxis] # (LNE, GD)
            #print("cell_edge_unit_tagnet:", cell_edge_unit_tagnet.shape, "\n", cell_edge_unit_tagnet)
            beta = np.einsum("ij, ij -> i", cell_edge_unit_tagnet, N)
            #print("beta:", beta.shape, "\n", beta)
            R = np.einsum('l, lg, l -> lg', beta, rot_distVec, cell_edge_measure) # (LNE, GD)
            #print("R:", R.shape, "\n", R)

            M_consistency = R @ np.linalg.inv(R.T @ N) @ R.T
            M_stability = np.trace(R @ R.T) / cell_measure[i] / LNE * \
                                              (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M = M_consistency + M_stability # (LNE, LNE)
            #print("M:", M.shape, "\n", M.round(3))

            lhs = np.einsum('en, n, ee, eg -> g', local_grad_h, p_ch, M, N) # (GD, )
            LHS.append(lhs)

            err = np.max(np.abs(M @ N - R))
            error.append(err)

            indexi, indexj = np.meshgrid(cell2edge[i], cell2edge[i])
            ME[indexi, indexj] += M

        return ME, np.array(LHS), np.array(error)

    def grad_operator(self):
        """
        离散梯度算子

        Results
        - result (ndarray, (NE, NN) )
        """
        mesh = self.mesh
        edge = mesh.entity('edge')
        NE = mesh.number_of_edges()
        NN = mesh.number_of_nodes()
        edge_measure = mesh.entity_measure(etype=1)

        gradh = np.zeros((NE, NN))
        for i in range(NE):
            gradh[i, edge[i, 0]] = -1 / edge_measure[i]
            gradh[i, edge[i, 1]] = 1 / edge_measure[i]

        return gradh


    def M_f(self):
        """
        全局 M_F 矩阵
        
        Results
        - result (ndarray, (NE, NE) )
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        cell2edge = mesh.ds.cell_to_edge()
        norm = mesh.edge_unit_normal()
        edge_centers = mesh.entity_barycenter(etype=1)
        cell_centers = mesh.entity_barycenter(etype=2) # (NC, GD)
        flag = np.where(mesh.ds.cell_to_edge_sign().toarray(), 1, -1) # (NC, NE)
        edge_measure = mesh.entity_measure(etype=1)
        cell_measure = mesh.entity_measure(etype=2)
        result = np.zeros((NE, NE))

        for i in range(NC):
            LNE = len(cell2edge[i])
            cell_out_flag = flag[i][cell2edge[i]] # (LNE, )
            cell_edge_measure = edge_measure[cell2edge[i]] # (LNE, )
            cell_edge_centers = edge_centers[cell2edge[i]] # (LNE, GD)

            R = np.einsum('l, lg, l -> lg', cell_out_flag, cell_edge_centers-cell_centers[i, :],
                          cell_edge_measure) # (LNE, GD)
            N = norm[cell2edge[i], :] # (LNE, GD)

            M_consistency = R @ np.linalg.inv(R.T @ N) @ R.T
            M_stability = np.trace(R @ R.T) / cell_measure[i] * \
                        (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            #M_stability = np.trace(M_consistency) / cell_measure[i] * \
            #            (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M = M_consistency + M_stability # (LNE, LNE)
            indexi, indexj = np.meshgrid(cell2edge[i], cell2edge[i])
            result[indexi, indexj] += M

        return result

    def M_c(self):
        """

        Results
        - result (ndarray, (NC, NC) )
        """
        mesh = self.mesh
        cell_measure = mesh.entity_measure("cell")
        result = np.diag(cell_measure)

        return result
    
    def div_operator(self):
        """
        离散散度算子

        Results
        - result (ndarray, (NC, NE) )
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        cell_measure = mesh.entity_measure(etype=2)
        edge_measure = mesh.entity_measure(etype=1)
        cell2edge = mesh.ds.cell_to_edge()
        flag = np.where(mesh.ds.cell_to_edge_sign().toarray(), 1, -1)
        result = np.zeros((NC, NE))
        for i in range(NC):
            cell_out_flag = flag[i][cell2edge[i]] # (LNE, )
            cell_edge_measure = edge_measure[cell2edge[i]] # (LNE, )
            result[i, cell2edge[i]] = cell_out_flag * cell_edge_measure/cell_measure[i]

        return result



    def source(self, fun, gddof, D):
        """

        Results
        - result (ndarray, (NC+NE, ) )
        """
        mesh = self.mesh
        NE = mesh.number_of_edges()
        cell_centers = mesh.entity_barycenter(etype=2)
        cell_measure = mesh.entity_measure('cell') 
        
        edge_measure = mesh.entity_measure('edge') 
        edge_centers = mesh.entity_barycenter(etype=1)

        f = fun(cell_centers)
        b = cell_measure * f

        gamma = np.zeros(NE)
        gamma[gddof] = edge_measure[gddof] * D(edge_centers[gddof])
        result = np.hstack((-gamma, -b))

        return result


    def u_M_f(self, velocity, ac=None):
        """

        Results
        - result (ndarray, (NE, NE) )
        """
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')
        #print("cell:", cell)
        #print("edge:", edge)
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        cell2edge = mesh.ds.cell_to_edge()
        #print("cell2edge:", cell2edge)
        cell2node = mesh.ds.cell_to_node()
        #print("cell2node:", cell2node)
        norm = mesh.edge_unit_normal()
        edge_centers = mesh.entity_barycenter(etype=1)
        cell_centers = mesh.entity_barycenter(etype=2) # (NC, GD)
        flag = np.where(mesh.ds.cell_to_edge_sign().toarray(), 1, -1)
        #print("flag:", flag.shape, "\n", flag)
        edge_measure = mesh.entity_measure(etype=1)
        cell_measure = mesh.entity_measure(etype=2)
        edge_norm = mesh.edge_unit_normal()
        #print("edge_norm:", edge_norm.shape, "\n", edge_norm)
        result = np.zeros((NE, NE))
        if ac is None:
            ac = cell_measure

        u_nodes = velocity(node)
        #print("u_nodes:", u_nodes.shape, "\n", u_nodes)
        u_edges = (u_nodes[edge[:, 0]] + u_nodes[edge[:, 1]]) / 2
        #print("u_edges:", u_edges.shape, "\n", u_edges)

        result = []
        for i in range(NC):
            edge_c = edge[cell[i]] # (LNE, 2)
            #print("edge_c:", edge_c.shape, "\n", edge_c)
            u_edges_c = (u_nodes[edge_c[:, 0]] + u_nodes[edge_c[:, 1]]) / 2 # (LNE, 2)
            #print("u_edges_c:", u_edges_c)
            tmp1 = edge_norm[cell2edge[i]]
            #print("tmp1:", tmp1)
            tmp2 = flag[i, cell2edge[i]].reshape(-1, 1)
            #print("tmp2:", tmp2)
            edge_norm_c = tmp1 * tmp2
            #print("edge_norm_c:", edge_norm_c) # (LNE, 2)
            LNE = len(cell2edge[i])
            cell_out_flag = flag[i][cell2edge[i]] # (LNE, )
            #print("cell_out_flag:", cell_out_flag)
            cell_edge_measure = edge_measure[cell2edge[i]] # (LNE, )
            cell_edge_centers = edge_centers[cell2edge[i]] # (LNE, GD)

            R = np.einsum('l, lg, l -> lg', cell_out_flag, cell_edge_centers-cell_centers[i, :],
                          cell_edge_measure) # (LNE, GD)
            N = norm[cell2edge[i], :] # (LNE, GD)

            M_consistency = R @ np.linalg.inv(R.T @ N) @ R.T
            #M_stability = np.trace(R@R.T) /cell_measure[i] * (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M_stability = cell_measure[i] * (np.eye(LNE) - N @ np.linalg.inv(N.T @ N) @ N.T)
            M = M_consistency + M_stability # (LNE, LNE)

            uh_c = np.einsum('ij, ij -> i', u_edges_c, edge_norm_c) # (LNE, )
            #print("uh_c:", uh_c) # (LNE, )

            uhc_M = M @ uh_c # (LNE, )

            result.append(uhc_M)

        #print("result:", result)
        UM = np.zeros((NC, NE))
        for i, (cell_edges, uhc_M) in enumerate(zip(cell2edge, result)):
            UM[i, cell_edges] = uhc_M

        return UM

    
    

    def source_neumann(self, fun):
        """

        Results
        - result (ndarray, (NC+NE, ) )
        """
        mesh = self.mesh
        NE = mesh.number_of_edges()
        cell_centers = mesh.entity_barycenter(etype=2)
        cell_measure = mesh.entity_measure('cell') 

        f = fun(cell_centers)
        b = cell_measure * f

        gamma = np.zeros(NE)
        result = np.hstack((-gamma, -b))

        return result

