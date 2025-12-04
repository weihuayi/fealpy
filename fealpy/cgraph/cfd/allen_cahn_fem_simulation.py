from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['AllenCahnFEMSimulation']

class AllenCahnFEMSimulation(CNodeType):
    r"""Finite Element Discretization for the Allen-Cahn Equation.

    This node constructs the finite element bilinear and linear forms
    required for the discretization of the Allen-Cahn equation using the provided function space.

    Inputs:
        epsilon (float): Interface thickness parameter.
        gamma (float): Mobility parameter.
        phispace (SpaceType): Finite element space for the phase field φ.
        init_phase (FunctionType): Initial phase field function.
        q (int): Quadrature degree used for numerical integration.
    """
    TITLE: str = "Allen-Cahn 方程有限元离散"
    PATH: str = "simulation.discretization"
    DESC: str = """
                使用有限元方法对 Allen-Cahn 方程进行离散。
                本节点实现了相场方程的有限元双线性形式与线性形式组装，适用于相场变量 φ 与化学势 μ 的耦
                合系统。通过输入界面厚度参数、迁移率参数、相场函数空间 以及积分精度，自动构造
                相应的双线性形式与线性形式，并生成一个用于时间步推进的更新函数 
                update(u_n, phi_n, dt, phase_force , mv), mv 为移动网格产生的速度场，可选。
                在每个时间步调用时返回当前的离散矩阵 A 与右端项 b。
                """
    INPUT_SLOTS = [
        PortConf("epsilon", DataType.FLOAT, title="界面厚度参数"),
        PortConf("gamma", DataType.FLOAT, title="迁移率参数"),
        PortConf("init_phase", DataType.FUNCTION, title="初始相场函数"),
        PortConf("phispace", DataType.SPACE, title="相场函数空间"),
        PortConf("q", DataType.INT, 0, default = 4, min_val=3, title="积分精度"),
    ]
    OUTPUT_SLOTS = [
        PortConf("update_ac", DataType.FUNCTION, title="更新函数")
    ]
    @staticmethod
    def run(epsilon, gamma, init_phase, phispace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import (ScalarMassIntegrator, ScalarConvectionIntegrator,
                                SourceIntegrator,ScalarDiffusionIntegrator)
        from fealpy.fem import (BilinearForm, LinearForm, BlockForm)
        from fealpy.sparse import COOTensor
        
        mesh = phispace.mesh
        init_mass = mesh.integral(init_phase)
        
        def update_ac(u_n, phi_n, dt,phase_force, mv: None):
            # 移动网格行为产生的速度场 mv 可选
            if mv is None:
                def mv(bcs):
                    NC = mesh.number_of_cells()
                    GD = mesh.geo_dimension()
                    shape = (NC, bcs.shape[0], GD)
                    return bm.zeros(shape, dtype=bm.float64)
                
            bform = BilinearForm(phispace)
            lform = LinearForm(phispace)
            
            SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/epsilon**2), q=q)
            SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q)
            SCI = ScalarConvectionIntegrator(q=q)
            SSI = SourceIntegrator(q=q)
            
            bform.add_integrator(SMI, SDI, SCI)
            lform.add_integrator(SSI)
            
            @barycentric
            def convection_coef(bcs, index):
                result = dt * (u_n(bcs, index) - mv(bcs, index))
                return result
            
            def fphi(phi):
                tag0 = phi[:] > 1
                tag1 = (phi[:] >= -1) & (phi[:] <= 1)
                tag2 = phi[:] < -1
                f_val = phispace.function()
                f_val[tag0] = 2/epsilon**2  * (phi[tag0] - 1)
                f_val[tag1] = (phi[tag1]**3 - phi[tag1]) / (epsilon**2)
                f_val[tag2] = 2/epsilon**2 * (phi[tag2] + 1)
                return f_val
            
            @barycentric
            def source(bcs , index):
                phi_val = phi_n(bcs, index)  
                result = (1+ dt*gamma/epsilon**2) * phi_val
                result -= gamma * dt * fphi(phi_n)(bcs, index)
                ps = mesh.bc_to_point(bcs, index)
                result += dt * phase_force(ps)
                return result
            
            SCI.coef = convection_coef
            SSI.source = source
            A = bform.assembly()
            b = lform.assembly()

            def lagrange_multiplier(A, b):
                LagLinearForm = LinearForm(phispace)
                Lag_SSI = SourceIntegrator(source=1, q=q)
                LagLinearForm.add_integrator(Lag_SSI)
                LagA = LagLinearForm.assembly()
                A0 = -dt * gamma * bm.ones(phispace.number_of_global_dofs())
                A0 = COOTensor(bm.array([bm.arange(len(A0), dtype=bm.int32), 
                                        bm.zeros(len(A0), dtype=bm.int32)]), A0, spshape=(len(A0), 1))
                A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                        bm.arange(len(LagA), dtype=bm.int32)]), LagA,
                                        spshape=(1, len(LagA)))
                b0 = bm.array([init_mass], dtype=bm.float64)
                A_block = BlockForm([[A, A0], [A1, None]])
                A_block = A_block.assembly_sparse_matrix(format='csr')
                b_block = bm.concat([b, b0], axis=0)
                return A_block, b_block
            
            A_block, b_block = lagrange_multiplier(A, b)
            return A_block, b_block
        return update_ac
            
            