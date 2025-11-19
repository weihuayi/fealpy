from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["GearBox"]

class GearBox(CNodeType):
    
    TITLE: str = "变速箱（矩阵装配）"
    PATH: str = "有限元.方程离散"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, "网格"),
        PortConf("shaftmatrix", DataType.MENU, "轴系矩阵"),
        PortConf("space", DataType.SPACE, "函数空间"),
        PortConf("q", DataType.INT, title="积分精度", default=3, min_val=1, max_val=17),
    ]
    OUTPUT_SLOTS = [
        PortConf("stiffness", DataType.TENSOR, title="刚度矩阵S"),
        PortConf("mass", DataType.TENSOR, title="质量矩阵M"),
        PortConf("NS", DataType.TENSOR, title="自由度划分信息"),
        PortConf("G", DataType.TENSOR, title="耦合矩阵"),
        PortConf("mesh", DataType.MESH, "网格"),
    ]

    
    @staticmethod
    def rbe2_matrix(mesh):
        
        from ...backend import backend_manager as bm
        from ...sparse import coo_matrix

        NN = mesh.number_of_nodes()
        redges, rnodes = mesh.data.get_rbe2_edge()

        # the flag of nodes coupling with the shaft system 
        isCNode = mesh.data.get_node_data('isCNode')
        mesh.nodedata['isCNode'] = isCNode

        # the flag of reference nodes
        isRNode = bm.zeros(NN, dtype=bm.bool)
        isRNode = bm.set_at(isRNode, rnodes, True)

        # the flag of gearbox nodes, which are not reference nodes 
        isGBNode = bm.logical_not(isRNode)

        # the flag of surface nodes, which are the gearbox nodes that maybe 
        # constrained by the reference nodes (notice that some reference nodes
        # maybe not in the shaft system)
        isSNode = bm.zeros(NN, dtype=bm.bool)
        isSNode = bm.set_at(isSNode, redges[:, 1], True)

        # the flag of RBE2 nodes, which are the coupling nodes that are
        # constrained by the nodes in the shaft system.
        isRBE2 = isCNode[redges[:, 0]]
        isCSNode = bm.zeros(NN, dtype=bm.bool)
        isCSNode = bm.set_at(isCSNode, redges[isRBE2, 1], True)
        
        # the flag of fixed gear box nodes, which are the nodes that are fixed 
        # TODO: we should update the code to support genearal case of boundary
        # conditions
        name = mesh.data['boundary_conditions'][0][0]
        nset = mesh.data.get_node_set(name)
        isFNode = bm.zeros(NN, dtype=bm.bool)
        isFNode[nset] = True

        # add above flags to the mesh data 
        mesh.data.add_node_data('isRNode', isRNode)
        mesh.data.add_node_data('isGBNode', isGBNode)
        mesh.data.add_node_data('isSNode', isSNode)
        mesh.data.add_node_data('isCSNode', isCSNode)
        mesh.data.add_node_data('isFNode', isFNode)

        # put the flags into mesh.nodedata for visualization
        mesh.nodedata['isRNode'] = isRNode
        mesh.nodedata['isGBNode'] = isGBNode
        mesh.nodedata['isSNode'] = isSNode
        mesh.nodedata['isCSNode'] = isCSNode
        mesh.nodedata['isFNode'] = isFNode

        # construct the RBE2 matrix 
        ridx = bm.where(isCNode)[0]
        NC = ridx.shape[0] # number of coupling nodes

        sidx = bm.where(isCSNode)[0]
        NS = sidx.shape[0] # number of surface nodes

        rmap = bm.full(NN, -1, dtype=bm.int32)
        rmap = bm.set_at(rmap, ridx, bm.arange(NC, dtype=bm.int32))

        smap = bm.full(NN, -1, dtype=bm.int32)
        smap = bm.set_at(smap, sidx, bm.arange(NS, dtype=bm.int32))

        I = smap[redges[isRBE2, 1]]
        J = rmap[redges[isRBE2, 0]]

        assert bm.all(I >= 0) and bm.all(J >= 0), "RBE2: I/J 含有 -1，映射未覆盖所有耦合边"

        node = mesh.entity('node')
        v = node[redges[isRBE2, 0]] - node[redges[isRBE2, 1]]

        print(f"RBE2 matrix: {NS} surface nodes, {NC} reference nodes")

        kwargs = {'shape':(3*NS, 6*NC), 'itype': bm.int32, 'dtype': bm.float64}
        ones = bm.ones(NS, dtype=bm.float64)

        G  = coo_matrix((    ones, (3*I+0, 6*J+0)), **kwargs)
        G += coo_matrix((-v[:, 2], (3*I+0, 6*J+4)), **kwargs)
        G += coo_matrix(( v[:, 1], (3*I+0, 6*J+5)), **kwargs)

        G += coo_matrix((    ones, (3*I+1, 6*J+1)), **kwargs)
        G += coo_matrix(( v[:, 2], (3*I+1, 6*J+3)), **kwargs)
        G += coo_matrix((-v[:, 0], (3*I+1, 6*J+5)), **kwargs)

        G += coo_matrix((    ones, (3*I+2, 6*J+2)), **kwargs)
        G += coo_matrix((-v[:, 1], (3*I+2, 6*J+3)), **kwargs)
        G += coo_matrix(( v[:, 0], (3*I+2, 6*J+4)), **kwargs)
        G = G.coalesce() 
        G = G.tocsr().to_scipy()
        
        return G,mesh
    
    
    @staticmethod
    def shaft_linear_system(shaftmatrix, mesh):
        
        from ...backend import backend_manager as bm
        from scipy.sparse import block_diag, spdiags, csr_matrix

        S = shaftmatrix['stiffness_total_system']
        M = shaftmatrix['mass_total_system']
        layout = shaftmatrix['component_bearing_layout'][:, 1:]
        section = shaftmatrix['order_section_shaft']

        NN = int(section[1].sum())  

        # offset for the nodes in each shaft section 
        offset = [0] + [int(a) for a in bm.cumsum(section[1])]  

        nidmap = mesh.data['nidmap']
        # the mapping from the original node index to the natural number index
        imap = {f"{a[0]:.1f}" : a[1] for a in zip(section[0], range(section.shape[1]))}
        # the flag of the nodes in the shaft system which coupling with the
        # gearbox shell model
        isCNode = bm.zeros(NN, dtype=bm.bool)
        for i, j in layout[:, :2]:
            idx = offset[imap[f"{i:.1f}"]] + int(j) - 1
            isCNode[idx] = True

        isCDof = bm.repeat(isCNode, 6)

        S0 = S[bm.logical_not(isCDof), :]
        S1 = S[isCDof, :]

        S00 = S0[:, bm.logical_not(isCDof)]
        S01 = S0[:, isCDof]
        S11 = S1[:, isCDof]

        M0 = M[bm.logical_not(isCDof), :]
        M1 = M[isCDof, :]

        M00 = M0[:, bm.logical_not(isCDof)]
        M01 = M0[:, isCDof]
        M11 = M1[:, isCDof]


        # the flag of the coupling nodes in the gearbox shell model
        cnode = nidmap[layout[:, -1].astype(bm.int32)]
        isCNode = bm.zeros(mesh.number_of_nodes(), dtype=bm.bool)
        isCNode = bm.set_at(isCNode, cnode, True)
        mesh.data.add_node_data('isCNode', isCNode)

        # resort the columns of S01, S11, M01, M11 according to the cnode order
        re = bm.argsort(cnode)
        idx = bm.arange(6 * cnode.shape[0], dtype=bm.int32).reshape(-1, 6)
        idx = idx[re, :].flatten()

        S01 = csr_matrix(S01[:, idx])
        S11 = csr_matrix(S11[idx, :][:, idx])
        M01 = csr_matrix(M01[:, idx])
        M11 = csr_matrix(M11[idx, :][:, idx])

        B = shaftmatrix['bearing_stiffness_matrix']
        nb = B.shape[0]//6
        B = [B[i*6:(i+1)*6, 1:] for i in range(nb)]
        BS = block_diag(B, format='csr')

        BM = shaftmatrix['bearing_mass_matrix'].diagonal()[isCDof][idx]
        BM = spdiags(BM, 0, BM.shape[0], BM.shape[0], format='csr')

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]], BS, BM, mesh

    

    @staticmethod
    def box_linear_system(mesh, space,G):
        
        from ...fem import BilinearForm
        from ...fem import LinearElasticityIntegrator
        from ...fem import ScalarMassIntegrator as MassIntegrator
        from ...backend import backend_manager as bm
        from ...material import LinearElasticMaterial

        """
        Construct the linear system for the gearbox shell model.
        """
        for name, data in mesh.data['materials'].items():
            material = LinearElasticMaterial(name, **data) 

        bform = BilinearForm(space)
        
        integrator = LinearElasticityIntegrator(material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(space)
        integrator = MassIntegrator(material.density)
        bform.add_integrator(integrator)
        M = bform.assembly()

        S = S.to_scipy()
        M = M.to_scipy()

        S = (S + S.T)/2.0
        M = (M + M.T)/2.0


        # the flag of free nodes 
        isRNode = mesh.data.get_node_data('isRNode')
        isFNode = mesh.data.get_node_data('isFNode')
        isCSNode = mesh.data.get_node_data('isCSNode')
        isFreeNode = ~(isRNode | isFNode | isCSNode) 
        mesh.data.add_node_data('isFreeNode', isFreeNode)

        # the flag of free dofs
        isFreeDof = bm.repeat(isFreeNode, 3)
        isCSDof = bm.repeat(isCSNode, 3)

        S0 = S[isFreeDof, :]
        S1 = S[isCSDof, :]

        S00 = S0[:, isFreeDof]
        S01 = S0[:, isCSDof] @ G
        S11 = S1[:, isCSDof] @ G 
        S11 = G.T @ S11  


        M0 = M[isFreeDof, :]
        M1 = M[isCSDof, :]

        M00 = M0[:, isFreeDof]
        M01 = M0[:, isCSDof] @ G
        M11 = M1[:, isCSDof] @ G
        M11 = G.T @ M11 

        #self._check_symmetry_spd(S11, M11)

        S11 = (S11 + S11.T)/2.0  # ensure symmetry
        M11 = (M11 + M11.T)/2.0  # ensure symmetry

        return [[S00, S01], [S01.T, S11]], [[M00, M01], [M01.T, M11]] 
    
    @staticmethod
    def run(mesh, shaftmatrix, space, q):
        from scipy.sparse import bmat
        S0, M0, BS, BM, mesh = GearBox.shaft_linear_system(shaftmatrix, mesh)
        G,mesh = GearBox.rbe2_matrix(mesh)
        S1, M1 = GearBox.box_linear_system(mesh, space,G)
        
        N0 = S0[0][0].shape[0]  # number of free dofs in the shaft system
        N1 = S0[1][1].shape[0]  # number of coupling dofs
        N2 = S1[0][0].shape[0]  # number of free dofs in the gearbox shell model
        N3 = S1[1][1].shape[0]  # number of coupling dofs

        S = bmat([[S0[0][0],     S0[0][1],    None,         None],
                  [S0[1][0],  S0[1][1]+BS,    None,          -BS],
                  [    None,         None, S1[0][0],    S1[0][1]],
                  [    None,          -BS, S1[1][0], S1[1][1]+BS]]).tocsr()
        
        M = bmat([[M0[0][0],     M0[0][1],           None,     None],
                  [M0[1][0],     M0[1][1]+BM,        None,     None],
                  [    None,         None,    M1[0][0],    M1[0][1]],
                  [    None,         None,    M1[1][0], M1[1][1]+BM]]).tocsr()
        
        return S, M, [N0, N1, N2, N3], G,mesh