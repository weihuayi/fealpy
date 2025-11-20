import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("FlowPastFoil")
mesher = cgraph.create("NACA4Mesh2d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dbc_u = cgraph.create("ProjectDBC")
dbc_p = cgraph.create("ProjectDBC")
simulation = cgraph.create("IncompressibleNSIPCS")
timeline = cgraph.create("CFDTimeline")
IncompressibleNSRun = cgraph.create("IncompressibleNSIPCSRun")

pde(
    mu = 0.001,
    rho = 1.0,
    inflow = 2.0,
    box = [-0.5, 2.7, -0.4, 0.4]
)
mesher(
    m = 0.0,
    p = 0.0,
    t = 0.12,
    c = 1.0,
    alpha = 5.0,
    N = 100,
    box = pde().domain,
    h = 0.05
)
uspacer(mesh = mesher().mesh, p=2, gd = 2)
pspacer(mesh = mesher().mesh, p=1)
dbc_u(
    space = uspacer(), 
    dirichlet = pde().velocity_dirichlet, 
    is_boundary = pde().is_velocity_boundary
)
dbc_p(
    space = pspacer(), 
    dirichlet = pde().pressure_dirichlet, 
    is_boundary = pde().is_pressure_boundary
)
simulation(
    constitutive = 1,
    mu = pde().mu,
    rho = pde().rho,
    source = pde().source,
    uspace = uspacer(),
    pspace = pspacer(),
    is_pressure_boundary = pde().is_pressure_boundary,
    apply_bcu = dbc_u().apply_bc,
    apply_bcp = dbc_p().apply_bc,
    q = 3
) 
timeline(
    T0 = 0.0,
    T1 = 1.0,
    NT = 10000
)
IncompressibleNSRun(
    T0=timeline().T0,
    T1=timeline().T1,
    NL=timeline().NL,
    uspace = uspacer(), 
    pspace = pspacer(), 
    velocity_0 = pde().velocity_0,
    pressure_0 = pde().pressure_0,
    is_pressure_boundary = pde().is_pressure_boundary,
    predict_velocity = simulation().predict_velocity,
    correct_pressure = simulation().correct_pressure,
    correct_velocity = simulation().correct_velocity,
    mesh = mesher().mesh,
    output_dir = "/home/libz/naca"
)

WORLD_GRAPH.output(uh = IncompressibleNSRun().uh)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

