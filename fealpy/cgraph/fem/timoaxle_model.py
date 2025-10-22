from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle"]


class Timoaxle(CNodeType):
    r"""Train Wheel-Axle Finite Element Model Node.
    Inputs:
        space (SPACE): Scalar function space, e.g., Lagrange function space.
        beam_material (FUNCTION): Timoshenko beam material object.
        axle_material (FUNCTION): Axle material object.
        external_load (FUNCTION): Returns the global load vector.
        dirichlet_idx (FUNCTION): Returns Dirichlet degrees of freedom indices.

    Attributes:
        boundary_type (MENU): Type of boundary condition.
            Options:
                - fixed_left: Fixed at left end
                - simply_supported: Simply supported
                - custom: Custom constraints (requires extension in code)
        load_type (MENU): Type of applied load.
            Options:
                - axial_force: Axial force
                - bending_moment: Bending moment
                - distributed_load: Distributed load
        load_value (FLOAT): Magnitude of the applied load (unit depends on load type)
        submesh_index (TENSOR): Element indices handled by this integrator.
        coord_transform (TENSOR): Local-to-global coordinate transformation matrix.
        penalty (FLOAT): Penalty factor for Dirichlet boundary condition enforcement (default: 1e20).

    Outputs:
        K (TENSOR): Global stiffness matrix including boundary condition treatment.
        F (TENSOR): Global load vector including boundary condition effects.
    
    """
    TITLE: str = "列车轮轴有限元模型"
    PATH: str = "有限元.方程离散"
    DESC: str = "该节点将几何模型、网格与材料信息整合，定义边界条件与载荷，生成完整的有限元模型对象"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="标量函数空间", title="拉格朗日函数空间"),
        PortConf("beam_material", DataType.FUNCTION, 1, desc="Timoshenko 梁材料对象", title="梁材料对象"),
        PortConf("axle_material", DataType.FUNCTION, 1, desc="轮轴材料对象",title="轮轴材料对象"),
        PortConf("cell_index", DataType.TENSOR, 1, desc="本积分子负责的单元索引", title="单元索引范围"),
        PortConf("external_load", DataType.FUNCTION, 1, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_idx", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度索引", title="边界自由度索引"),
        
        PortConf("boundary_type", DataType.MENU, 0, desc="约束条件类型选择",title="边界条件类型",
                 items=["fixed_left", "simply_supported", "custom"]),
        PortConf("load_type", DataType.MENU, 0, desc="载荷类型选择",title="载荷类型", 
                 items=["axial_force", "bending_moment", "distributed_load"]),
        PortConf("load_value", DataType.FLOAT, 0, desc="施加载荷数值", title="载荷大小"),
        PortConf("penalty", DataType.FLOAT, 0, desc="乘大数法处理边界", title="系数", default=1e20)
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.TENSOR, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件作用的全局载荷向量",  title="全局载荷向量")
    ]

    @staticmethod
    def run(space, beam_material, axle_material, cell_index=None,
        external_load=None, dirichlet_idx=None,
        boundary_type=None, load_type=None, load_value=None, penalty=1e20):
        
        from fealpy.backend import bm
        from fealpy.sparse import COOTensor
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
        from fealpy.csm.fem.axle_integrator import AxleIntegrator

        tspace = TensorFunctionSpace(space, shape=(-1, 6))

        Dofs = tspace.number_of_global_dofs()
        K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
        F = bm.zeros(Dofs, dtype=bm.float64)

        timo_integrator = TimoshenkoBeamIntegrator(tspace, beam_material, 
                                index=bm.arange(0, cell_index-10))
        KE_beam = timo_integrator.assembly(tspace)
        ele_dofs_beam = timo_integrator.to_global_dof(tspace)

        for i, dof in enumerate(ele_dofs_beam):
                K[dof[:, None], dof] += KE_beam[i]

        axle_integrator = AxleIntegrator(tspace, axle_material, 
                                index=bm.arange(cell_index-10, cell_index))
        KE_axle = axle_integrator.assembly(tspace)
        ele_dofs_axle = axle_integrator.to_global_dof(tspace)   

        for i, dof in enumerate(ele_dofs_axle):
                K[dof[:, None], dof] += KE_axle[i]

        F[:] = external_load()
        
        penalty = penalty
        fixed_dofs = bm.asarray(dirichlet_idx, dtype=int)
        
        F[fixed_dofs] *= penalty
        for dof in fixed_dofs:
                K[dof, dof] *= penalty
                
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
        
        return K, F