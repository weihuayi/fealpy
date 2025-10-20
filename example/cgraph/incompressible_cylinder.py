import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("IncompressibleCylinder2d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dbc_u = cgraph.create("ProjectDBC")
dbc_p = cgraph.create("ProjectDBC")
simulation = cgraph.create("IncompressibleNSIPCS")
timeline = cgraph.create("CFDTimeline")
IncompressibleNSRun = cgraph.create("IncompressibleNSIPCSRun")
show = cgraph.create("OutputVideo")

pde(
    mu = 0.001,
    rho = 1.0,
    cx = 0.5,
    cy = 0.2,
    radius = 0.07,
    n_circle = 100,
    h = 0.06)
uspacer(mesh = pde().mesh, p=2, gd = 2)
pspacer(mesh = pde().mesh, p=1)
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
    q = 2
) 
timeline(
    T0 = 0.0,
    T1 = 0.2,
    NT = 200
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
    mesh = pde().mesh
)
show(
    T0 = timeline().T0,
    T1 = timeline().T1,
    NL = timeline().NL,
    domain = pde().domain,
    mesh = pde().mesh,  
    out = IncompressibleNSRun().uh_x,
    dpi = 300,
    bitrate = 3000,
    figsize_x = 6.0,
    figsize_y = 3.0,
    cmap = 'cividis',
    clim_vmin = 0.0,
    clim_vmax = 1.8,
    # filename = "Cylinder_04",
    title = "velocity_x"
)

WORLD_GRAPH.output(out = show().out)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

