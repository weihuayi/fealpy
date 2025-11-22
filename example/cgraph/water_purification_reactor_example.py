import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("WaterPurificationReactorMesher")
imesher = cgraph.create("Int1d")
wpr = cgraph.create("MGTensorWPR")
solver = cgraph.create("MGStokesSolver")

mesher(lc = 0.4)
imesher(interval=[0, 0.4], nx=8)
wpr(
    mesh=mesher().mesh,
    imesh=imesher().mesh,
    thickness=0.4
)
solver(
    op=wpr().op,
    F=wpr().F,
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

g1 = cgraph.Graph()
g1.output(uh=solver())
g1.register_error_hook(lambda x: print(x.traceback))
g1.execute()
result = g1.get()
uh = result["uh"]

print(uh)
