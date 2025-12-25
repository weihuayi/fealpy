import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("WaterPurificationReactorMesher")
wpr = cgraph.create("MGTensorWPR")
solver = cgraph.create("MGStokesSolver")
to_vtk = cgraph.create("TO_VTK")

mesher(lc = 0.4, nx=8)
wpr(
    tmesh=mesher().tmesh,
    imesh=mesher().imesh,
    thickness=0.4
)
solver(
    mesh=wpr().mesh,
    op=wpr().op,
    idx=wpr().idx,
    F1=wpr().F1,
    bd_flag=wpr().bd_flag,
    ugdof=wpr().ugdof,
    Ai=wpr().Ai,
    Bi=wpr().Bi,
    Bti=wpr().Bti,
    bigAi=wpr().bigAi,
    P_u=wpr().P_u,
    R_u=wpr().R_u,
    P_p=wpr().P_p,
    R_p=wpr().R_p,
    Nu=wpr().Nu,
    Np=wpr().Np,
    level=wpr().level,
    auxMat=wpr().auxMat,
    options=wpr().options
)

to_vtk(mesh = solver().mesh,
        uh = (solver().uh, solver().ph),
        path = "/home/zjx/py")

WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

