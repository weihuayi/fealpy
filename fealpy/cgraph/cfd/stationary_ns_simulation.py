
from ..nodetype import CNodeType, PortConf, DataType


__all__ = ["StationaryNS2d"]


class StationaryNSSimulation(CNodeType):
    TITLE: str = "StationaryNSSimulation"
    PATH: str = "cfd.simulation"
    INPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT),
        PortConf("rho", DataType.FLOAT),
        PortConf("source", DataType.FUNCTION),
        PortConf("uspace", DataType.SPACE),
        PortConf("pspace", DataType.SPACE),
        PortConf("q", DataType.INT, 0, default = 3)
    ]
    OUTPUT_SLOTS = [
        PortConf("BForm", DataType.LINOPS),
        PortConf("LForm", DataType.LINOPS),
        PortConf("update", DataType.FUNCTION),
        PortConf("lagrange_multiplier", DataType.FUNCTION)
    ]
    @staticmethod
    def run(mu, rho, source, uspace, pspace, q):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import (ScalarMassIntegrator,ScalarConvectionIntegrator, PressWorkIntegrator, ScalarDiffusionIntegrator,
                                SourceIntegrator)
 
        A00 = BilinearForm(uspace)
        u_BM_netwon = ScalarMassIntegrator(q = q)
        u_BC = ScalarConvectionIntegrator(q = q)
        u_BVW = ScalarDiffusionIntegrator(q = q)
        
        A00.add_integrator(u_BM_netwon)
        A00.add_integrator(u_BC)
        A00.add_integrator(u_BVW)
        
        A01 = BilinearForm((pspace, uspace))
        u_BPW = PressWorkIntegrator(q = q)
        A01.add_integrator(u_BPW)
       
        A = BlockForm([[A00, A01], [A01.T, None]]) 

        L0 = LinearForm(uspace)
        u_LSI = SourceIntegrator(q = q)
        u_source_LSI = SourceIntegrator(q = q)
        L0.add_integrator(u_LSI) 
        L0.add_integrator(u_source_LSI)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        
        def update(u0): 
            cv = mu
            cc = rho
            pc = 1.0
            cbf = source
            
            ## BilinearForm
            u_BVW.coef = cv
            u_BPW.coef = -pc

            @barycentric
            def u_BC_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                return cccoef * u0(bcs, index)
            u_BC.coef = u_BC_coef

            @barycentric
            def u_BM_netwon_coef(bcs,index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                return cccoef * u0.grad_value(bcs, index)
            u_BM_netwon.coef = u_BM_netwon_coef

            ## LinearForm 
            @barycentric
            def u_LSI_coef(bcs, index):
                cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result = cccoef*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
                return result
            u_LSI.source = u_LSI_coef
            u_source_LSI.source = cbf

        def lagrange_multiplier(A, b, c=0):
            """
            Constructs the augmented system matrix for Lagrange multipliers.
            c is the integral of pressure, default is 0.
            """
            from fealpy.sparse import COOTensor
            LagLinearForm = LinearForm(pspace)
            LagLinearForm.add_integrator(SourceIntegrator(source=1))
            LagA = LagLinearForm.assembly()
            LagA = bm.concatenate([bm.zeros(uspace.number_of_global_dofs()), LagA], axis=0)

            A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                    bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

            A = BlockForm([[A, A1.T], [A1, None]])
            A = A.assembly_sparse_matrix(format='csr')
            b0 = bm.array([c])
            b  = bm.concatenate([b, b0], axis=0)
            return A, b
        return (A, L, update, lagrange_multiplier)

    
