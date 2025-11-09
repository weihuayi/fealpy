from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss"]


class Truss(CNodeType):
    r"""
    Assembles the global stiffness matrix and load vector for a 3D truss (bar) finite element model.

    This node takes a scalar function space, material properties (E, A), an external load function, 
    and a function for Dirichlet boundary indices. It applies the boundary conditions and returns 
    the final system matrix `K` and vector `F`.

    Inputs:
        space (Space): A scalar Lagrange function space.
        bar_E (float): Elastic modulus of the bar material.
        A (float): Cross-sectional area of the bars.
        external_load (function): A function that returns the global external load vector.
        dirichlet_idx (function): A function that returns the indices of the degrees of freedom with Dirichlet boundary conditions.

    Outputs:
        K (tensor): The global stiffness matrix with boundary conditions applied.
        F (tensor): The global load vector with boundary conditions applied.
    """
    TITLE: str = "桁架有限元模型"
    PATH: str = "有限元.方程离散"
    DESC: str = "组装桁架系统的全局刚度矩阵和载荷向量，并处理边界条件"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("bar_E", DataType.FLOAT, 1, desc="杆的材料属性",  title="杆的弹性模量"),
        PortConf("A", DataType.TENSOR, 1, desc="横截面积",  title="杆的横截面积"),

        PortConf("external_load", DataType.FUNCTION, 1, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_idx", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度索引", title="边界自由度索引"),
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件作用的全局载荷向量",  title="全局载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        from fealpy.fem import BilinearForm, DirichletBC
        
        class material:
            def __init__(self, options: dict):
                self.E  = options.get("bar_E")
                self.A = options.get("A")
                
        bar_material = material(options=options) 
        
        space = options.get("space")
        external_load = options.get("external_load") 
        dirichlet_idx = options.get("dirichlet_idx") 
        
        tspace = TensorFunctionSpace(space, shape=(-1, 3))
        
        bform = BilinearForm(tspace)
        bform.add_integrator(BarIntegrator(space=tspace, material=bar_material)) 
        K = bform.assembly()

        F = external_load(space.mesh) 
        
        def zero_bc(p):
            return bm.zeros_like(p)

        fixed_dofs = dirichlet_idx(space.mesh)

        threshold = bm.zeros(tspace.number_of_global_dofs(), dtype=bool)
        threshold[fixed_dofs] = True
        
        bc = DirichletBC(space=tspace, gd=zero_bc, threshold=threshold)
        K, F = bc.apply(K, F)
        
        return K, F