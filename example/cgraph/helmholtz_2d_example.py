import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("Helmholtz2d")
mesher = cgraph.create("Box2d")
spacer = cgraph.create("FunctionSpace")
isDDof = cgraph.create("BoundaryDof")
uher = cgraph.create("FEFunction")
helmholtz_eq = cgraph.create("HelmholtzEquation")
dbc = cgraph.create("DirichletBC")
solver = cgraph.create("CGSolver")


# 连接节点
spacer(mesh=mesher(), p=1)
helmholtz_eq(
    space=spacer(),
    q=3,
    diffusion= 1,
    reaction = -1,
    source=pde().source
)
dbc(
    gd=pde().dirichlet,
    isDDof=isDDof(space=spacer()),
    form=helmholtz_eq().operator,
    F=helmholtz_eq().source
)
solver(
    A=dbc().A,
    b=dbc().F,
    x0=dbc().uh
)
# 最终连接到图输出节点上
WORLD_GRAPH.output_node(mesh=mesher(), uh=solver())

WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())