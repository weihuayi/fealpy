from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Timoaxle"]


class Timoaxle(CNodeType):
    r"""Assemble the global stiffness matrix and load vector for the train axle finite element model.
    
    Inputs:
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        GD (INT): Geometric dimension of the model.
        space (Space): Scalar function space.
        beam_E (float): Elastic modulus of the beam component.
        beam_nu (float): Poisson's ratio of the beam component.
        axle_E (float): Elastic modulus of the axle (shaft) component.
        axle_nu (float): Poisson's ratio of the axle (shaft) component.
        cindex (int): Total number of elements in the beam–axle model.
        external_load (function): Function that returns the global load vector.
        dirichlet_dof (function): Function that returns the Dirichlet boundary degree-of-freedom.
        penalty (float, optional): Penalty coefficient used for enforcing Dirichlet boundary conditions.
        
    Outputs:
        K (tensor): Global stiffness matrix after applying boundary constraints.
        F (tensor): Global load vector after applying the selected load type and boundary conditions.
    
    """
    TITLE: str = "列车轮轴有限元模型"
    PATH: str = "simulation.discretization"
    DESC: str = """"该节点基于前处理阶段给定的几何参数与材料参数，构建列车轮轴的梁-杆耦合有限元模型。
            调用对应材料模型计算单元刚度矩阵。随后，将所有单元刚度汇总组装为全局刚度矩阵，
            并通过罚函数法施加 Dirichlet 边界条件。输出包含边界约束后的全局刚度矩阵与全局载荷向量。"""
            
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁材料属性",  title="梁的泊松比"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的弹性模量"),
        PortConf("axle_nu", DataType.FLOAT, 1, desc="弹簧材料属性",  title="弹簧的泊松比"),
        PortConf("external_load", DataType.FUNCTION, 1, desc="返回全局载荷向量", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度索引", title="边界自由度索引"),
        
        PortConf("cindex", DataType.INT, 0, desc="轮轴模型的单元单元总数", title="单元总数", default=32),
        PortConf("penalty", DataType.FLOAT, 0, desc="乘大数法处理边界", title="系数", default=1e20),
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件作用的全局载荷向量",  title="全局载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.sparse import COOTensor
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.fem.axle_integrator import AxleIntegrator
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator

        model = TimobeamAxleData3D(
            beam_para=options.get("beam_para"),
            axle_para=options.get("axle_para")
        )

        beam_E  = options.get("beam_E")
        beam_nu = options.get("beam_nu")
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timoshenko_beam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)

        axle_E  = options.get("axle_E")
        axle_nu = options.get("axle_nu")
        axle_material = BarMaterial(model=model,
                                name="Bar",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)

        GD = options.get("GD")
        space = options.get("space")
        cindex = options.get("cindex")
        external_load = options.get("external_load")
        dirichlet_dof = options.get("dirichlet_dof")
        penalty = options.get("penalty")
        
        tspace = TensorFunctionSpace(space, shape=(-1, GD*2))

        Dofs = tspace.number_of_global_dofs()
        K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
        F = bm.zeros(Dofs, dtype=bm.float64)
        
        timo_integrator = TimoshenkoBeamIntegrator(tspace, 
                                    model=model,
                                    material=beam_material, 
                                    index=bm.arange(0, cindex-10))
        KE_beam = timo_integrator.assembly(tspace)
        ele_dofs_beam = timo_integrator.to_global_dof(tspace)

        for i, dof in enumerate(ele_dofs_beam):
                K[dof[:, None], dof] += KE_beam[i]

        axle_integrator = AxleIntegrator(tspace, 
                                model=model,
                                material=axle_material,
                                index=bm.arange(cindex -10, cindex ))
        KE_axle = axle_integrator.assembly(tspace)
        ele_dofs_axle = axle_integrator.to_global_dof(tspace)   

        for i, dof in enumerate(ele_dofs_axle):
                K[dof[:, None], dof] += KE_axle[i]

        penalty = penalty
        fixed_dofs = bm.asarray(dirichlet_dof, dtype=int)

        F = external_load
        F[fixed_dofs] *= penalty
        
        for dof in fixed_dofs:
                K[dof, dof] *= penalty
                
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
        
        return K, F