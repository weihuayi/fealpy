
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNSRun(CNodeType):
    TITLE: str = "StationaryNSRun"
    PATH: str = "cfd.run"
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, default=1000, min_val=1),
        PortConf("tol", DataType.FLOAT, default=1e-6, min_val=1e-10, max_val=1e-2),
        PortConf("update", DataType.FUNCTION),
        PortConf("apply_bc", DataType.FUNCTION),
        PortConf("lagrange_multiplier", DataType.FUNCTION),
        PortConf("A", DataType.LINOPS),
        PortConf("L", DataType.LINOPS),
        PortConf("uspace", DataType.SPACE),
        PortConf("pspace", DataType.SPACE),
        PortConf("mesh", DataType.MESH),
    ]
    OUTPUT_SLOTS = [
        PortConf("uh1", DataType.TENSOR),
        PortConf("ph1", DataType.TENSOR)
    ]
    @staticmethod
    def run(maxstep, tol, update, apply_bc, lagrange_multiplier, A, L, uspace, pspace, mesh):
        from fealpy.solver import spsolve
        print(1)
        uh0 = uspace.function()
        ph0 = pspace.function()
        uh1 = uspace.function()
        ph1 = pspace.function()
        ugdof = uspace.number_of_global_dofs()
        for i in range(maxstep):
            update(uh0)
            A = A.assembly()
            F = L.assembly()
            A, F = apply_bc(A, L)
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
        return (uh1.array, ph1.array)