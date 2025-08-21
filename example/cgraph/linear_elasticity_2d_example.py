import json
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("LinearElasticity2d")
mesher = cgraph.create("Box2d")
spacer = cgraph.create("TensorFunctionSpace")
isDDof = cgraph.create("BoundaryDof")
uher = cgraph.create("FEFunction")
linear_elasticity_eq = cgraph.create("LinearElasticityEquation")
dbc = cgraph.create("DirichletBC")
solver = cgraph.create("CGSolver")

# 连接节点
spacer(mesh=mesher(), p=1)
linear_elasticity_eq(
    space=spacer(),
    q=3,
    lam=pde().lam,
    mu=pde().mu,
    hypo=pde().hypo,
    body_force=pde().body_force,
)
dbc(
    gd=pde().displacement_bc,
    isDDof=isDDof(space=spacer()),
    form=linear_elasticity_eq().operator,
    F=linear_elasticity_eq().body_force
)
solver(
    A=dbc().A,
    b=dbc().F,
    x0=dbc().uh
)

# 最终连接到图输出节点上
WORLD_GRAPH.output_node(mesh=mesher(), uh=solver())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())