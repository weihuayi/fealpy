import json
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("TimobeamAxle3d")
mesher = cgraph.create("CreateMesh")
spacer = cgraph.create("TensorFunctionSpace")
isDDof = cgraph.create("BoundaryDof")
uher = cgraph.create("FEFunction")
timobeam_axle = cgraph.create("TimobeamAxle")
dbc = cgraph.create("DirichletBC")
solver = cgraph.create("CGSolver")

# 连接节点
spacer(mesh=mesher(), p=1)
timobeam_axle(
    space=spacer(),
    q=3,
    lam=pde().lam,
    mu=pde().mu,
    hypo=pde().hypo,
    external_load=pde().external_load,
)
dbc(
    gd=pde().displacement_bc,
    isDDof=isDDof(space=spacer()),
    form=timobeam_axle().operator,
    F=timobeam_axle().external_load
)
solver(
    A=dbc().A,
    b=dbc().F,
    x0=dbc().uh
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher(), uh=solver())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())