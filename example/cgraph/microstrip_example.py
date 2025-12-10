
from fealpy import logger
import fealpy.cgraph as cgraph

logger.setLevel("INFO")
mesher = cgraph.create("MicrostripPatchMesher3d")
msa = cgraph.create("MicrostripAntenna3D")
solver = cgraph.create("CGSolver")

msa(
    mesh=mesher().mesh,
    sub_region=mesher().sub,
    air_region=mesher().air,
    pec_face=mesher().pec,
    lumped_edge=mesher().lumped,
    r0=80.0,
    r1=100.0
)
solver(
    A=msa().operator,
    b=msa().vector,
    x0 = msa().uh
)


g1 = cgraph.Graph()
g1.output(uh=solver())
g1.register_error_hook(lambda x: print(x.traceback))
g1.execute()
result = g1.get()
uh = result["uh"]

print(uh)
