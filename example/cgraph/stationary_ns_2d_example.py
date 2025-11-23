
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("StationaryNS2d")
mesher = cgraph.create("Box2d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
simulation = cgraph.create("StationaryNSNewton")
dbc = cgraph.create("StationaryNSDBC")
StationaryNSRun = cgraph.create("StationaryNSRun")

mesher(domain = pde().domain, nx = 10, ny = 10)
uspacer(mesh = mesher(), p=2, gd = 2)
pspacer(mesh = mesher(), p=1)
simulation(
    constitutive = 1,
    mu = pde().mu,
    rho = pde().rho,
    source = pde().source,
    uspace = uspacer(),
    pspace = pspacer(),
    q = 2
) 
dbc(
    uspace = uspacer(), 
    pspace = pspacer(), 
    velocity_dirichlet = pde().velocity_dirichlet, 
    pressure_dirichlet = pde().pressure_dirichlet, 
    is_velocity_boundary = pde().is_velocity_boundary, 
    is_pressure_boundary = pde().is_pressure_boundary
)
StationaryNSRun(
    maxstep=1000,
    tol=1e-6,
    update = simulation().update,
    apply_bc = dbc().apply_bc,
    BForm = simulation().BForm,
    LForm = simulation().LForm,
    uspace = uspacer(), 
    pspace = pspacer(), 
    mesh = mesher()
)

WORLD_GRAPH.output(uh = StationaryNSRun().uh, ph = StationaryNSRun().ph, 
                   uh_x = StationaryNSRun().uh_x, uh_y = StationaryNSRun().uh_y)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
