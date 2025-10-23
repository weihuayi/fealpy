from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle"]


class Timoaxle(CNodeType):
    r"""Train Wheel-Axle Finite Element Model Node.
    Inputs:
        space (SPACE): Scalar function space, e.g., Lagrange function space.
        beam_material (FUNCTION): Timoshenko beam material object.
        axle_material (FUNCTION): Axle material object.
        external_load (TENSON): Returns the global load vector.
        dirichlet_idx (TENSON): Returns Dirichlet degrees of freedom indices.

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
    DESC: str = "组装轮轴系统的刚度矩阵和载荷"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="标量函数空间", title="拉格朗日函数空间"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="弹性模量",  title="梁的材料属性"),
        PortConf("beam_mu", DataType.FLOAT, 1, desc="剪切模量",  title="梁的材料属性"),
        PortConf("Ax", DataType.FLOAT, 1, desc="X 轴横截面积",  title="横截面积"),
        PortConf("Ay", DataType.FLOAT, 1, desc="Y 轴的横截面积",  title="横截面积"),
        PortConf("Az", DataType.FLOAT, 1, desc="Z 轴的横截面积",  title="横截面积"),
        PortConf("J", DataType.FLOAT, 1, desc="X 轴的惯性矩",  title="极性矩"),
        PortConf("Iy", DataType.FLOAT, 1, desc="Y 轴的惯性矩",  title="惯性矩"),
        PortConf("Iz", DataType.FLOAT, 1, desc="Z 轴的惯性矩",  title="惯性矩"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="弹性模量",  title="杆的材料属性"),
        PortConf("axle_mu", DataType.FLOAT, 1, desc="剪切模量",  title="杆的材料属性"),
        
        PortConf("cindex", DataType.INT, 0, desc="轮轴模型的单元单元总数", title="单元总数"),
        PortConf("external_load", DataType.FUNCTION, 1, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_idx", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度索引", title="边界自由度索引"),
        
        PortConf("penalty", DataType.FLOAT, 0, desc="乘大数法处理边界", title="系数", default=1e20),
        PortConf("boundary_type", DataType.MENU, 0, desc="约束条件类型选择",title="边界条件类型", default="fixed",
                 items=["fixed", "simply_supported", "custom"]),
        PortConf("load_type", DataType.MENU, 0, desc="载荷类型选择",title="载荷类型", default="force",
                 items=["force", "bending", "distributed_load"])
    ]
    OUTPUT_SLOTS = [
         PortConf("boundary_type", DataType.MENU, desc="约束条件类型",title="边界条件类型"),
        PortConf("load_type", DataType.MENU, desc="载荷类型",title="载荷类型"),
        PortConf("K", DataType.TENSOR, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件作用的全局载荷向量",  title="全局载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.sparse import COOTensor
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.fem.axle_integrator import AxleIntegrator
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
        
        class Bmaterial:
            def __init__(self, options: dict):
                self.E  = options.get("beam_E")
                self.mu = options.get("beam_mu")
                self.Ax = options.get("Ax")
                self.Ay = options.get("Ay")
                self.Az = options.get("Az")
                self.J  = options.get("J")
                self.Iy = options.get("Iy")
                self.Iz = options.get("Iz")
                
        
        beam_material = Bmaterial(options=options) 
        
        class Amaterial:
            def __init__(self, options: dict):
                self.E  = options.get("axle_E")
                self.mu = options.get("axle_mu")
        
        axle_material = Amaterial(options=options) 
        
        space = options.get("space")
        cindex = options.get("cindex")
        external_load = options.get("external_load")
        dirichlet_idx = options.get("dirichlet_idx")
        penalty = options.get("penalty")
        boundary_type = options.get("boundary_type")
        load_type = options.get("load_type")
        
        
        tspace = TensorFunctionSpace(space, shape=(-1, 6))

        Dofs = tspace.number_of_global_dofs()
        K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
        F = bm.zeros(Dofs, dtype=bm.float64)
        
        timo_integrator = TimoshenkoBeamIntegrator(tspace, beam_material, 
                                index=bm.arange(0, cindex-10))
        KE_beam = timo_integrator.assembly(tspace)
        ele_dofs_beam = timo_integrator.to_global_dof(tspace)

        for i, dof in enumerate(ele_dofs_beam):
                K[dof[:, None], dof] += KE_beam[i]

        axle_integrator = AxleIntegrator(tspace, axle_material, 
                                index=bm.arange(cindex -10, cindex ))
        KE_axle = axle_integrator.assembly(tspace)
        ele_dofs_axle = axle_integrator.to_global_dof(tspace)   

        for i, dof in enumerate(ele_dofs_axle):
                K[dof[:, None], dof] += KE_axle[i]

        penalty = penalty
        fixed_dofs = bm.asarray(dirichlet_idx(), dtype=int)
        
        F = external_load()
        F[fixed_dofs] *= penalty
        
        for dof in fixed_dofs:
                K[dof, dof] *= penalty
                
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
        
        return boundary_type, load_type, K, F