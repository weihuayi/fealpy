from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss"]


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