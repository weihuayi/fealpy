from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle"]


class Timoaxle(CNodeType):
    r"""Assemble the global stiffness matrix and load vector for the train axle finite element model.
    
    Inputs:
        space (Space): Scalar function space.
        beam_E (float): Elastic modulus of the beam component.
        beam_mu (float): Shear modulus of the beam component.
        Ax (float): Cross-sectional area in the X direction.
        Ay (float): Cross-sectional area in the Y direction.
        Az (float): Cross-sectional area in the Z direction.
        J (float): Polar moment of inertia of the beam cross-section.
        Iy (float): Second moment of inertia about the Y axis.
        Iz (float): Second moment of inertia about the Z axis.
        axle_E (float): Elastic modulus of the axle (shaft) component.
        axle_mu (float): Shear modulus of the axle (shaft) component.
        cindex (int): Total number of elements in the beam–axle model.
        external_load (function): Function that returns the global load vector.
        dirichlet_idx (function): Function that returns the Dirichlet boundary degree-of-freedom indices.
        penalty (float, optional): Penalty coefficient used for enforcing Dirichlet boundary conditions.
        boundary_type (str, optional): Type of boundary constraint to apply.
        load_type (str, optional): Type of external load to apply.
        
    Outputs:
        boundary_type (str): Applied boundary constraint type.
        load_type (str): Applied load type.
        K (tensor): Global stiffness matrix after applying boundary constraints.
        F (tensor): Global load vector after applying the selected load type and boundary conditions.
    
    """
    TITLE: str = "列车轮轴有限元模型"
    PATH: str = "有限元.方程离散"
    DESC: str = "组装轮轴系统的刚度矩阵和载荷"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的弹性模量"),
        PortConf("beam_mu", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的剪切模量"),
        PortConf("Ax", DataType.TENSOR, 1, desc="横截面积",  title="X 方向的横截面积"),
        PortConf("Ay", DataType.TENSOR, 1, desc="横截面积",  title="Y 方向的横截面积"),
        PortConf("Az", DataType.TENSOR, 1, desc="横截面积",  title="Z 方向的横截面积"),
        PortConf("J", DataType.TENSOR, 1, desc="极性矩",  title="极性矩"),
        PortConf("Iy", DataType.TENSOR, 1, desc="惯性矩",  title="Y 轴的惯性矩"),
        PortConf("Iz", DataType.TENSOR, 1, desc="惯性矩",  title="Z 轴的惯性矩"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的弹性模量"),
        PortConf("axle_mu", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的剪切模量"),
        
        PortConf("cindex", DataType.INT, 0, desc="轮轴模型的单元单元总数", title="单元总数", default=32),
        PortConf("external_load", DataType.FUNCTION, 1, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_idx", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度索引", title="边界自由度索引"),
        
        PortConf("penalty", DataType.FLOAT, 0, desc="乘大数法处理边界", title="系数", default=1e20),
        PortConf("boundary_type", DataType.MENU, 0, desc="约束条件类型选择",title="边界条件类型", default="fixed",
                 items=["fixed", "simply_supported", "custom"]),
        PortConf("load_type", DataType.MENU, 0, desc="载荷类型选择",title="载荷类型", default="force",
                 items=["force", "bending", "distributed_load"])
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
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
        
        return K, F