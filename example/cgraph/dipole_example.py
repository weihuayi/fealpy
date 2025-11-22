
import json
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("DipoleAntenna3D")
mesher = cgraph.create("Dipole3d")
spacer = cgraph.create("FunctionSpace")
uher = cgraph.create("FEFunction")
dipole_antenna_eq = cgraph.create("DipoleAntennaEquation")
dbc = cgraph.create("DirichletBC")
solver = cgraph.create("IterativeSolver")
pross = cgraph.create("AntennaPostprocess")


# 连接节点
mesher(cr = 0.05, sr0 = 1.9, sr1 = 2.4, L = 2.01, G = 0.01)
spacer(type = "first_nedelec", mesh=mesher().mesh, p=1)
dipole_antenna_eq(
    space=spacer(),
    q=3,
    diffusion = pde().diffusion,
    reaction = pde().reaction,
    source = pde().source,
    Y = pde().Y,
    ID = mesher().ID1
)

dbc(
    gd=pde().dirichlet,
    isDDof=mesher().ID2,
    form=dipole_antenna_eq().operator,
    F=dipole_antenna_eq().source
)

solver(
    A=dbc().A,
    b=dbc().F,
    solver = "minres"
)

pross(uh = solver().out, space = spacer())

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher().mesh, uh=solver(), E =pross())

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

