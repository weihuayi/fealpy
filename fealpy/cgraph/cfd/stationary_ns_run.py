
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNSRun(CNodeType):
    TITLE: str = "StationaryNSRun"
    PATH: str = "cfd.run"
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, default=1000, min_val=1),
        PortConf("tol", DataType.FLOAT, default=1e-6, min_val=1e-10, max_val=1e-2),
        PortConf("simulation", DataType.NONE),
        PortConf("pde", DataType.NONE),

        
    ]
    OUTPUT_SLOTS = [
        PortConf("uh1", DataType.FUNCTION),
        PortConf("ph1", DataType.FUNCTION)
    ]
    @staticmethod
    def run(maxstep, simulation, tol, pde):
        from fealpy.solver import spsolve
        from fealpy.fem.dirichlet_bc import DirichletBC
        
        uspace = simulation.uspace
        pspace = simulation.pspace
        BForm = simulation.BForm()
        LForm = simulation.LForm()
        update = simulation.update
        uh0 = uspace.function()
        ph0 = pspace.function()
        uh1 = uspace.function()
        ph1 = pspace.function()
        ugdof = uspace.number_of_global_dofs()
        
        for i in range(maxstep):
            update(uh0)
            A = BForm.assembly()
            F = LForm.assembly()
            
            BC = DirichletBC(
                (uspace, pspace), 
                gd=(pde.velocity, pde.pressure), 
                threshold=(pde.is_velocity_boundary, pde.is_pressure_boundary), 
                method='interp')
            
            A, F = BC.apply(A, F)
            A, F = simulation.lagrange_multiplier(A, F)
            x = spsolve(A, F,"mumps")
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:-1]
            res_u = simulation.mesh.error(uh0, uh1)
            res_p = simulation.mesh.error(ph0, ph1)
            
            if res_u + res_p < tol:
                break
            uh0[:] = uh1
            ph0[:] = ph1
        return uh1,ph1