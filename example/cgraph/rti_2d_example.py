import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')
bm.set_default_device('cpu')


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("RayleighTaylor")
mesher = cgraph.create("Box2d")
phispacer = cgraph.create("FunctionSpace")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
bdf2 = cgraph.create("IncompressibleNSBDF2")
chfem = cgraph.create("CahnHilliardFEMSimulation")
chnsrun = cgraph.create("CHNSFEMRun")

mesher(
    mesh_type="triangle",
    domain=pde().domain,
    nx=64,
    ny=256
)
phispacer(
    mesh=mesher().mesh,
    p = 2
)
uspacer(
    mesh=mesher().mesh,
    p = 2,
    gd = 2
)
pspacer(
    mesh=mesher().mesh,
    p = 1
)
bdf2(
    Re = pde().Re,
    uspace = uspacer(), 
    pspace = pspacer(),
    q = 3
)
chfem(
    epsilon = pde().epsilon,
    Pe = pde().Pe,
    phispace = phispacer(),
    q = 5,
    s = 1.0
)
chnsrun(
    dt = 0.00125*bm.sqrt(bm.array(2)),
    nt = 2000,
    rho_up = pde().rho_up,
    rho_down = pde().rho_down,
    Fr = pde().Fr,
    ns_update = bdf2().update,
    ch_update = chfem().update,
    phispace = phispacer(),
    uspace = uspacer(),
    pspace = pspacer(),
    mesh = mesher(),
    init_interface = pde().init_interface,
    is_ux_boundary = pde().is_ux_boundary,
    is_uy_boundary = pde().is_uy_boundary
)


WORLD_GRAPH.output(uh = chnsrun().u)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

