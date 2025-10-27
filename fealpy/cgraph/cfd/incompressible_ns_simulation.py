from ..nodetype import CNodeType, PortConf, DataType

def lagrange_multiplier(pspace, A, b, c=0):
    """
    Constructs the augmented system matrix for Lagrange multipliers.
    c is the integral of pressure, default is 0.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.sparse import COOTensor
    from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
    LagLinearForm = LinearForm(pspace)
    LagLinearForm.add_integrator(SourceIntegrator(source=1))
    LagA = LagLinearForm.assembly()

    A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                            bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b0 = bm.array([c])
    b  = bm.concatenate([b, b0], axis=0)
    return A, b

class IncompressibleNSIPCS(CNodeType):
    r"""Unsteady Incompressible Navier-Stokes solver using the IPCS algorithm.

    Inputs:
        constitutive (int): Constitutive relation type (1: Newtonian, 2: generalized).
        mu (float): Dynamic viscosity.
        rho (float): Fluid density.
        source (function): Source term (external force) as a function of space and time.
        uspace(space): Function space for the velocity field.
        pspace(space): Function space for the pressure field.
        is_pressure_boundary (function): Predicate function for identifying pressure boundaries.
        apply_bcu (function): Function to apply velocity boundary conditions.
        apply_bcp (function): Function to apply pressure boundary conditions.
        q (int): Quadrature order for numerical integration (default: 3).

    Outputs:
        predict_velocity (function): Function that assembles the predicted velocity system.
        correct_pressure (function): Function that assembles the pressure correction system.
        correct_velocity (function): Function that assembles the velocity correction system.
    """
    TITLE: str = "非稳态 NS 方程 IPCS 算法"
    PATH: str = "流体.有限元算法"
    DESC: str = """该节点基于有限元法实现不可压 Navier-Stokes 方程的非稳态求解，采用 IPCS 分步算法
                构建预测速度、压力修正和速度修正三个离散系统，支持多种本构模型与边界条件设置。"""
    INPUT_SLOTS = [
        PortConf("constitutive", DataType.MENU, 0, title="本构方程", default=1, items=[i for i in range(1, 2)]),
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界"),
        PortConf("apply_bcu", DataType.FUNCTION, title="速度边界处理函数"),
        PortConf("apply_bcp", DataType.FUNCTION, title="压力边界处理函数"),
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度")
    ]
    OUTPUT_SLOTS = [
        PortConf("predict_velocity", DataType.FUNCTION, title="预测速度方程离散"),
        PortConf("correct_pressure", DataType.FUNCTION, title="压力修正方程离散"),
        PortConf("correct_velocity", DataType.FUNCTION, title="速度修正方程离散")
    ]
    @staticmethod
    def run(constitutive, mu, rho, source, uspace, pspace, is_pressure_boundary,
            apply_bcu, apply_bcp, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.backend import TensorLike
        from fealpy.decorator import barycentric, cartesian
        from fealpy.fem import (BilinearForm, ScalarMassIntegrator, ScalarDiffusionIntegrator,
                                ViscousWorkIntegrator, FluidBoundaryFrictionIntegrator, DirichletBC)
        #预测速度左端项
        predict_Bform = BilinearForm(uspace)
        predict_BM = ScalarMassIntegrator(q=q)  
        predict_Bform.add_integrator(predict_BM)
        
        if is_pressure_boundary() != 0:
            predict_BF = FluidBoundaryFrictionIntegrator(q=q, threshold=is_pressure_boundary)
            predict_Bform.add_integrator(predict_BF)
        
        if constitutive == 1:
            predict_BVW = ScalarDiffusionIntegrator(q=q)
            predict_Bform.add_integrator(predict_BVW)
        elif constitutive == 2:
            predict_BVW = ViscousWorkIntegrator(q=q)
            predict_Bform.add_integrator(predict_BVW)

        #预测速度右端项
        from fealpy.fem import (LinearForm, SourceIntegrator, GradSourceIntegrator, 
                                BoundaryFaceSourceIntegrator)
        predict_Lform = LinearForm(uspace) 
        predict_LS = SourceIntegrator(q=q)
        predict_LS_f = SourceIntegrator(q=q)
        predict_LGS = GradSourceIntegrator(q=q)
        
        predict_Lform.add_integrator(predict_LS)
        predict_Lform.add_integrator(predict_LGS)
        predict_Lform.add_integrator(predict_LS_f)
        if is_pressure_boundary() != 0:
            predict_LBFS = BoundaryFaceSourceIntegrator(q=q, threshold=is_pressure_boundary)
            predict_Lform.add_integrator(predict_LBFS)

        #预测速度更新函数
        def predict_velocity_update(u0, p0, t, dt): 
            mesh = uspace.mesh
            ctd = rho 
            cv = mu
            cc = rho
            pc = 1
            cbf = cartesian(lambda p:source(p, t))
            
            predict_BM.coef = ctd/dt
            predict_BVW.coef = cv
            
            @barycentric
            def LS_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result = 1/dt*masscoef*u0(bcs, index)
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result -= ccoef * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
                return result

            @barycentric
            def LGS_coef(bcs, index):
                I = bm.eye(mesh.GD)
                result = bm.repeat(p0(bcs,index)[...,bm.newaxis], mesh.GD, axis=-1)
                result = bm.expand_dims(result, axis=-1) * I
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            
            
            @barycentric
            def LBFS_coef(bcs, index):
                result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), mesh.face_unit_normal(index=index))
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            
            predict_LS_f.source = cbf
            predict_LS.source = LS_coef
            predict_LGS.source = LGS_coef
            if is_pressure_boundary() != 0:
                predict_BF.coef = -cv
                predict_LBFS.source = LBFS_coef

        #预测速度方程线性系统组装
        def predict_velocity(u0:TensorLike, p0:TensorLike, t:float, dt:float): 
            Bform = predict_Bform
            Lform = predict_Lform
            predict_velocity_update(u0, p0, t, dt)
            A = Bform.assembly()
            b = Lform.assembly()
            apply = apply_bcu(t)
            A, b = apply(A, b)
            return A, b 
            
        #压力修正左端项
        pressure_Bform = BilinearForm(pspace)
        pressure_BD = ScalarDiffusionIntegrator(q=q)
        pressure_Bform.add_integrator(pressure_BD) 

        #压力修正右端项
        pressure_Lform = LinearForm(pspace)
        pressure_LS = SourceIntegrator(q=q)
        pressure_LGS = GradSourceIntegrator(q=q)
        
        pressure_Lform.add_integrator(pressure_LS)
        pressure_Lform.add_integrator(pressure_LGS)

        #压力修正更新函数
        def pressure_update(us, p0, dt):
            pc = 1
            ctd = rho
            
            pressure_BD.coef = pc

            @barycentric
            def LS_coef(bcs, index=None):
                result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
                result *= ctd(bcs, index) if callable(ctd) else ctd
                return result
            pressure_LS.source = LS_coef
            
            @barycentric
            def LGS_coef(bcs, index=None):
                result = p0.grad_value(bcs, index)
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            pressure_LGS.source = LGS_coef

        #压力修正方程线性系统组装
        def correct_pressure(us:TensorLike, p0:TensorLike, t, dt:float):
            Bform = pressure_Bform
            Lform = pressure_Lform
            pressure_update(us, p0, dt)
            A = Bform.assembly()
            b = Lform.assembly()
            if is_pressure_boundary() == 0:
                A, b = lagrange_multiplier(pspace, A, b, 0)
            else:
                apply = apply_bcp(t)
                A, b = apply(A, b)
            return A, b
            
        #速度修正左端项
        correct_Bform = BilinearForm(uspace)
        correct_BM = ScalarMassIntegrator(q=q)
        correct_Bform.add_integrator(correct_BM)

        #速度修正右端项
        correct_Lform = LinearForm(uspace)
        correct_LS = SourceIntegrator(q=q)
        correct_Lform.add_integrator(correct_LS)

        #速度修正更新函数
        def correct_velocity_update(us, p0, p1, dt):
            ctd = rho 
            cp = 1
            correct_BM.coef = ctd
            @barycentric
            def BM_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result = masscoef * us(bcs, index)
                result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
                return result
            correct_LS.source = BM_coef

        #速度修正方程线性系统组装
        def correct_velocity(us:TensorLike, p0:TensorLike, p1:TensorLike, 
                             t, dt:float, ):
            """速度校正"""
            Bform = correct_Bform
            Lform = correct_Lform
            correct_velocity_update(us, p0, p1, dt)
            A = Bform.assembly()
            b = Lform.assembly()
            apply = apply_bcu(t)
            A, b = apply(A, b)
            return A, b
            
        return predict_velocity, correct_pressure, correct_velocity