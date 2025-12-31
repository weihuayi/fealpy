from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['BoundaryCondition', 'BeamBoundaryCondition']


class BoundaryCondition(CNodeType):
    r"""Apply Dirichlet boundary conditions for bar elements.

    Supports two methods:
    1. Direct Method: Directly modifies stiffness matrix (better conditioning)
    2. Penalty Method: Multiplies diagonal terms by large penalty factor
    
    Inputs:
        mesh (MESH): Mesh object containing bar elements.
        K (LINOPS): Global stiffness matrix in CSR format.
        method (STRING): Boundary condition method ("direct" or "penalty").
        penalty (FLOAT): Penalty factor for penalty method (default: 1e12).
        
    Outputs:
        K_bc (LINOPS): Modified stiffness matrix with boundary conditions applied.
        F_bc (TENSOR): Modified load vector with boundary conditions applied.

    Note:
        - mesh.nodedata['load']: Applied loads at nodes (NN, 3).
        - mesh.nodedata['constraint']: Constraint flags (NN, 4), format [node_idx, flag_x, flag_y, flag_z].
        - Direct method maintains better numerical conditioning than Penalty method.
        - Constrained DOF displacements are set to 0 (can be extended for non-zero prescribed values).
    """
    TITLE: str = "杆单元边界条件处理"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="包含杆单元的网格对象",  
                 title="网格"),
        PortConf("K", DataType.LINOPS, 1, 
                 desc="全局刚度矩阵 (稀疏CSR格式)", 
                 title="刚度矩阵"),
        PortConf("method", DataType.MENU, 0, 
                 desc="边界条件处理方法", 
                 title="处理方法",
                 default="direct",
                 items=["direct", "penalty"]),
        PortConf("penalty", DataType.FLOAT, 0, 
                 desc="惩罚系数 (仅用于乘大数法,建议值: 1e12)", 
                 title="惩罚系数", 
                 default=1e12)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K_bc", DataType.LINOPS, 
                 desc="应用边界条件后的刚度矩阵", 
                 title="边界处理后的刚度矩阵"),
        PortConf("F_bc", DataType.TENSOR, 
                 desc="应用边界条件后的载荷向量", 
                 title="边界处理后的载荷向量")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.sparse import CSRTensor
        
        mesh = options.get("mesh")
        K = options.get("K")
        method = options.get("method")
        penalty = options.get("penalty")
        
        NN = mesh.number_of_nodes()
        load = mesh.nodedata['load'] 
        constraint = mesh.nodedata['constraint'] 
        
        F_bc = load.flatten()
        K_dense = K.toarray()
    
        node_indices = constraint[:, 0].astype(bm.int32)
        constraint_flags = constraint[:, 1:4]
        
        # 构建节点自由度映射
        node_dofs = bm.zeros((NN, 3), dtype=bm.int32)
        for j in range(3):
            node_dofs[:, j] = 3 * node_indices + j

        # 找出所有被约束的自由度
        is_constrained = constraint_flags > 0.5
        constrained_dofs = node_dofs[is_constrained]
        
        if method == "direct":
            K_dense[constrained_dofs, :] = 0.0
            K_dense[:, constrained_dofs] = 0.0
            K_dense[constrained_dofs, constrained_dofs] = 1.0
            F_bc[constrained_dofs] = 0.0
            
        elif method == "penalty":
            K_dense[constrained_dofs, constrained_dofs] *= penalty
            F_bc[constrained_dofs] = 0.0
            
        else:
            raise ValueError(f"Unknown boundary condition method: {method}. Use 'direct' or 'penalty'.")
        
        # 转换回 CSR 稀疏格式
        rows, cols = bm.nonzero(K_dense)
        values = K_dense[rows, cols]
        
        crow = bm.zeros(K_dense.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K_bc = CSRTensor(crow, cols, values, spshape=K_dense.shape)
        
        return K_bc, F_bc
    

class BeamBoundaryCondition(CNodeType):
    r"""Apply Dirichlet boundary conditions for beam elements.
    
    Supports two methods for 6-DOF beam elements (u_x, u_y, u_z, θ_x, θ_y, θ_z):
    1. Direct Method: Directly modifies stiffness matrix (better conditioning)
    2. Penalty Method: Multiplies diagonal terms by large penalty factor
    
    Inputs:
        mesh (MESH): Mesh object containing beam elements.
        K (LINOPS): Global stiffness matrix in CSR format.
        method (STRING): Boundary condition method ("direct" or "penalty").
        penalty (FLOAT): Penalty factor for penalty method (default: 1e20).
        
    Outputs:
        K_bc (LINOPS): Modified stiffness matrix with boundary conditions applied.
        F_bc (TENSOR): Modified load vector with boundary conditions applied.
    """
    TITLE: str = "梁单元边界条件处理"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="包含梁单元的网格对象",  
                 title="网格"),
        PortConf("K", DataType.LINOPS, 1, 
                 desc="全局刚度矩阵 (稀疏CSR格式)", 
                 title="刚度矩阵"),
        PortConf("method", DataType.MENU, 0, 
                 desc="边界条件处理方法", 
                 title="处理方法",
                 default="penalty",
                 items=["direct", "penalty"]),
        PortConf("penalty", DataType.FLOAT, 0, 
                 desc="惩罚系数 (仅用于乘大数法,建议值: 1e20)", 
                 title="惩罚系数", 
                 default=1e20)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K_bc", DataType.LINOPS, 
                 desc="应用边界条件后的刚度矩阵", 
                 title="边界处理后的刚度矩阵"),
        PortConf("F_bc", DataType.TENSOR, 
                 desc="应用边界条件后的载荷向量", 
                 title="边界处理后的载荷向量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.sparse import CSRTensor
        
        mesh = options.get("mesh")
        K = options.get("K")
        method = options.get("method")
        penalty = options.get("penalty")
        
        NN = mesh.number_of_nodes()
        load = mesh.nodedata['load']  # (NN, 6)
        constraint = mesh.nodedata['constraint']  # (NN, 7)
        
        # 梁单元每个节点6个自由度
        node_ldof = 6
        
        F_bc = load.flatten()
        K_dense = K.toarray()
    
        node_indices = constraint[:, 0].astype(bm.int32)
        constraint_flags = constraint[:, 1:7]  

        # 构建节点自由度映射
        node_dofs = bm.zeros((NN, node_ldof), dtype=bm.int32)
        for j in range(node_ldof):
            node_dofs[:, j] = node_ldof * node_indices + j

        # 找出所有被约束的自由度
        is_constrained = constraint_flags > 0.5
        constrained_dofs = node_dofs[is_constrained]
        
        if method == "direct":
            K_dense[constrained_dofs, :] = 0.0
            K_dense[:, constrained_dofs] = 0.0
            K_dense[constrained_dofs, constrained_dofs] = 1.0
            F_bc[constrained_dofs] = 0.0
            
        elif method == "penalty":
            K_dense[constrained_dofs, constrained_dofs] *= penalty
            F_bc[constrained_dofs] = 0.0
            
        else:
            raise ValueError(f"Unknown boundary condition method: {method}. Use 'direct' or 'penalty'.")
        
        rows, cols = bm.nonzero(K_dense)
        values = K_dense[rows, cols]
        
        crow = bm.zeros(K_dense.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K_bc = CSRTensor(crow, cols, values, spshape=K_dense.shape)
        
        return K_bc, F_bc