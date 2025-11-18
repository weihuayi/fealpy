from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss", "TrussTower"]


class Truss(CNodeType):
    r"""
    Assembles the global stiffness matrix and load vector for a 3D truss (bar) finite element model.

    This node takes a function space, mesh, material properties (E, A), load parameters, 
    and fixed node indices. It constructs the stiffness matrix, generates the load vector, 
    applies Dirichlet boundary conditions, and returns the final system matrix `K` and vector `F`.

    Inputs:
        space (Space): A scalar Lagrange function space.
        mesh (Mesh): The mesh object representing the truss structure.
        bar_E (float): Elastic modulus of the bar material.
        A (float): Cross-sectional area of the bars.
        p (float): The magnitude of the concentrated force.
        top_z (float): The Z-coordinate of the nodes where the load is applied.
        fixed_nodes (tensor): A tensor containing the indices of nodes with fixed (zero displacement) boundary conditions.

    Outputs:
        K (tensor): The global stiffness matrix with boundary conditions applied.
        F (tensor): The global load vector with boundary conditions applied.
    """
    TITLE: str = "桁架有限元模型"
    PATH: str = "有限元.方程离散"
    DESC: str = "组装全局刚度矩阵K并应用边界条件, 输出K与F"

    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("bar_E", DataType.FLOAT, 1, desc="弹性模量", title="杆的弹性模量"),
        PortConf("A", DataType.FLOAT, 1, desc="横截面积", title="杆的横截面积"),
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("p", DataType.FLOAT, 0, desc="沿y轴的集中力", title="载荷", default=900.0),
        PortConf("top_z", DataType.FLOAT, 0, desc="受力层Z坐标", title="受力层Z", default=5080.0),
        PortConf("fixed_nodes", DataType.TENSOR, 1, desc="固定约束的节点", title="固定节点", default=[6, 7, 8, 9]),
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="处理边界后的刚度矩阵", title="全局刚度矩阵"),
        PortConf("F", DataType.TENSOR, desc="处理边界后的载荷向量", title="全局载荷向量"),
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        from fealpy.fem import BilinearForm, DirichletBC

        class material:
            def __init__(self, options: dict):
                self.E = options.get("bar_E")
                self.A = options.get("A")

        bar_material = material(options=options)
        space = options.get("space")
        mesh = options.get("mesh")
        top_z = options.get("top_z")
        p = options.get("p")
        fixed_nodes = options.get("fixed_nodes")
        
        node_coords = mesh.entity("node")
        GD, NN = 3, mesh.number_of_nodes()
        F = bm.zeros((NN, GD), dtype=bm.float64)
        idx = bm.where(node_coords[..., 2] == top_z)[0]
        if idx.size > 0:
            F[idx] = bm.array([0.0, p, 0.0], dtype=bm.float64)
        F = F.reshape(-1)

        fn = bm.asarray(fixed_nodes, dtype=bm.int32)
        dof_list = []
        for k in range(3):
            dof_list.append(3*fn + k)
        fixed_dofs = bm.concatenate(dof_list).astype(bm.int32)
        
        tspace = TensorFunctionSpace(space, shape=(-1, 3))
        bform = BilinearForm(tspace)
        bform.add_integrator(BarIntegrator(space=tspace, material=bar_material))
        K = bform.assembly()

        def zero_bc(p):
            return bm.zeros_like(p)

        threshold = bm.zeros(tspace.number_of_global_dofs(), dtype=bool)
        threshold[fixed_dofs] = True
        bc = DirichletBC(space=tspace, gd=zero_bc, threshold=threshold)
        K, F = bc.apply(K, F)
        return K, F
    

class TrussTower(CNodeType):
    r"""Truss Tower Finite Element Model Node.
    
    Inputs:
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        GD (INT): Geometric dimension of the model.
        space (Space): Lagrange function space for the truss tower model.
        E (float): Young's modulus of the bar elements (Pa).
        nu (float): Poisson's ratio of the bar elements.
        vertical (INT): Boolean flags indicating vertical columns.
        other (INT): Boolean flags indicating other bars.
        load (float): Total vertical load applied at top nodes.
        dirichlet_dof (Function): Dirichlet boundary DOFs.
        
    Outputs:
        K (LinearOperator): Global stiffness matrix with boundary conditions applied.
        F (Tensor): Global load vector with boundary conditions applied.
    
    """
    TITLE: str = "桁架塔有限元模型"
    PATH: str = "simulation.discretization"
    DESC: str = """"该节点基于给定的函数空间、材料参数、杆件分类信息与外部载荷，组装桁架塔的全局刚度矩阵与载荷向量，
            并施加 Dirichlet 边界条件，最终构建可直接用于力学求解的有限元离散系统。"""
            
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径"),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量",  title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比",  title="泊松比"),
        PortConf("load", DataType.TENSOR, 1, desc="全局载荷向量，表示总载荷如何分布到顶部节点", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, 1, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("vertical", DataType.INT, 0, desc="竖向杆件的个数",  title="竖向杆件", default=76),
        PortConf("other", DataType.INT, 0, desc="其他杆件的个数",  title="其他杆件", default=176)
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="算子",),
        PortConf("F", DataType.TENSOR, desc="含边界条件处理后的载荷向量",  title="载荷"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.sparse import CSRTensor
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.fem import DirichletBC
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        GD = options.get("GD")
        scalar_space = options.get("space")
        space = TensorFunctionSpace(scalar_space, shape=(-1, GD))
        
        gdof  = space.number_of_global_dofs()
        K = bm.zeros((gdof, gdof), dtype=bm.float64)

        # 获取立柱和斜杆的个数
        vertical = options.get("vertical")
        other = options.get("other")
        
        vertical_indices = bm.arange(0, vertical, dtype=bm.int32)
        other_indices = bm.arange(vertical, vertical + other, dtype=bm.int32)

        vertical_integrator = BarIntegrator(
                space=space,
                model=model,
                material=material,
                index=vertical_indices
            )
        KE_vertical = vertical_integrator.assembly(space) # (NC_v, ldof, ldof)
        ele_dofs_vertical = vertical_integrator.to_global_dof(space)  # (NC_v, ldof)
        
        for i in range(len(ele_dofs_vertical)):
            dof = ele_dofs_vertical[i]
            K[dof[:, None], dof] += KE_vertical[i]
        
        other_integrator = BarIntegrator(
                space=space,
                model=model,
                material=material,
                index=other_indices
            )
        KE_other = other_integrator.assembly(space)  # (NC_o, ldof, ldof)
        ele_dofs_other = other_integrator.to_global_dof(space)  # (NC_o, ldof)
        
        for i in range(len(ele_dofs_other)):
            dof = ele_dofs_other[i]
            K[dof[:, None], dof] += KE_other[i]

        F = options.get("load")

        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = options.get("dirichlet_dof")
        threshold[fixed_dofs] = True
        
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        crow = bm.zeros(K.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)

        K_sparse = CSRTensor(crow, cols, values, spshape=K.shape)

        bc = DirichletBC(
                space=space,
                gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                threshold=threshold
            )
        K, F = bc.apply(K_sparse, F)
        
        return K, F