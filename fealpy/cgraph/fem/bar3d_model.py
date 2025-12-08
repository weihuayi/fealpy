from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarModel", "TrussTower"]


class BarModel(CNodeType):
    r"""Assembles the global stiffness matrix and load vector for a 3D bar finite element model.

    Inputs:
        bar_type(MENU): Type of bar structure.
        space_type (str): Type of function space (e.g., "lagrangespace").
        GD (int): Geometric dimension of the model.
        mesh (mesh): A scalar Lagrange function space.
        E (float): Elastic modulus of the bar material.
        nu (float): Poisson's ratio of the bar material.
        external_load (float): The global load vector applied to the structure.
        dirichlet_dof (tensor): Boolean flags indicating Dirichlet boundary DOFs.
        dirichlet_bc (tensor): Prescribed displacement values for all DOFs.
        
    Outputs:
        K (linops): The global stiffness matrix with boundary conditions applied.
        F (tensor): The global load vector with boundary conditions applied.
    """
    TITLE: str = "杆件有限元模型"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("bar_type", DataType.MENU, 0, desc="杆件结构类型", title="杆件类型", 
                 default="bar25", items=["bar25", "bar942"]),
        PortConf("space_type", DataType.MENU, 0, title="函数空间类型", default="LagrangeFESpace", items=["lagrangespace"]),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("mesh", DataType.MESH, 1, desc="杆件网格", title="网格"),
        PortConf("E", DataType.FLOAT, 1, desc="杆的弹性模量", title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆的泊松比", title="泊松比"),
        PortConf("external_load", DataType.FLOAT, 1, desc="集中载荷", title="载荷"),
        PortConf("dirichlet_dof", DataType.TENSOR, 1, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("dirichlet_bc", DataType.TENSOR, 1, desc="边界节点的位移约束值", title="边界位移的值"),
        PortConf("penalty", DataType.FLOAT, 0, desc="乘大数法系数", title="乘大数法系数", default=1e12)

    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="处理边界后的刚度矩阵", title="算子"),
        PortConf("F", DataType.TENSOR, desc="处理边界后的载荷向量", title="载荷"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.sparse import CSRTensor
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm, DirichletBC
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        
        bar_type = options.get("bar_type")
        if bar_type == "bar25":
            from fealpy.csm.model.truss.bar_data25_3d import BarData25
            pde = BarData25()
        elif bar_type == "bar942":
            from fealpy.csm.model.truss.bar_data942_3d import BarData942
            pde = BarData942()
        else:
            raise ValueError(f"Unsupported bar_type: {bar_type}")
        
        mesh = options.get("mesh")
        space = LagrangeFESpace(mesh, p=1)
        GD = options.get("GD")
        tspace = TensorFunctionSpace(space, shape=(-1, GD))
        
        material = BarMaterial(
            model=pde,
            name=options.get("bar_type"),
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu"),
        )

        bform = BilinearForm(tspace)
        integrator = BarIntegrator(space=tspace, 
                                        model=pde, 
                                        material=material)
        bform.add_integrator(integrator)
        K = bform.assembly()
        F = options.get("external_load")
        
        is_bd_dof = options.get("dirichlet_dof")
        F = F.flatten()
        
        if bar_type == "bar25":
            # DirichletBC 方法
            gdof = tspace.number_of_global_dofs()
            threshold = bm.zeros(gdof, dtype=bool)
            threshold[is_bd_dof] = True
             
            gd_value = options.get("dirichlet_bc")
            bc = DirichletBC(space=tspace, gd=gd_value, threshold=threshold)
            K, F = bc.apply(K, F)
        elif bar_type == "bar942":
            # 乘大数法
            fixed_dofs = bm.where(is_bd_dof)[0]
            penalty = options.get("penalty")
            
            F[fixed_dofs] *= penalty
            
            K = K.toarray()
            for dof in fixed_dofs:
                K[dof, dof] *= penalty
            
            rows, cols = bm.nonzero(K)
            values = K[rows, cols]
            crow = bm.zeros(K.shape[0] + 1, dtype=bm.int64)
            for i in range(len(rows)):
                crow[rows[i] + 1] += 1
            crow = bm.cumsum(crow)
            
            K = CSRTensor(crow, cols, values, spshape=K.shape)
        
        return K, F
    
class TrussTower(CNodeType):
    r"""Truss Tower Finite Element Model Node.
    
    Inputs:
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        space_type (str): Type of function space (e.g., "lagrangespace").
        GD (int): Geometric dimension of the model.
        mesh (mesh): A scalar Lagrange function space.
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
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径"),
        PortConf("space_type", DataType.MENU, 0, title="函数空间类型", default="LagrangeFESpace", items=["lagrangespace"]),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("mesh", DataType.MESH, 1, desc="桁架塔网格", title="网格"),
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
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
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
        mesh = options.get("mesh")
        scalar_space = LagrangeFESpace(mesh, p=1)
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