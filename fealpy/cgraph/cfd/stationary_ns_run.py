
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNSRun(CNodeType):
    TITLE: str = "StationaryNSRun"
    PATH: str = "cfd.run"
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, 0, default=1000, min_val=1),
        PortConf("tol", DataType.FLOAT, 0, default=1e-6, min_val=1e-12, max_val=1e-2),
        PortConf("update", DataType.FUNCTION),
        PortConf("apply_bc", DataType.FUNCTION),
        PortConf("lagrange_multiplier", DataType.FUNCTION),
        PortConf("BForm", DataType.LINOPS),
        PortConf("LForm", DataType.LINOPS),
        PortConf("uspace", DataType.SPACE),
        PortConf("pspace", DataType.SPACE),
        PortConf("mesh", DataType.MESH),
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.FUNCTION),
        PortConf("ph", DataType.FUNCTION)
    ]
    @staticmethod
    def run(maxstep, tol, update, apply_bc, lagrange_multiplier, BForm, LForm, uspace, pspace, mesh):
        from fealpy.solver import spsolve
        uh0 = uspace.function()
        ph0 = pspace.function()
        uh1 = uspace.function()
        ph1 = pspace.function()
        ugdof = uspace.number_of_global_dofs()
        for i in range(maxstep):
            update(uh0)
            A = BForm.assembly()
            F = LForm.assembly()
            A, F = apply_bc(A, F)
            A, F = lagrange_multiplier(A, F)
            x = spsolve(A, F,"mumps")
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:-1]
            res_u = mesh.error(uh0, uh1)
            res_p = mesh.error(ph0, ph1)
            
            if res_u + res_p < tol:
                break
            uh0[:] = uh1
            ph0[:] = ph1

        # from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.exp0001 import Exp0001
        # pde = Exp0001()
        # error_u = mesh.error(uh1, pde.velocity)
        # error_p = mesh.error(ph1, pde.pressure)
        # print(f"Final: error_u = {error_u}, error_p = {error_p}")
        
        return uh1, ph1