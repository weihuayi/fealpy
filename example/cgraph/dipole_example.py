
import json
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("DipoleAntenna3D")
mesher = cgraph.create("Dipole3d")
dipole_antenna_eq = cgraph.create("DipoleAntennaEquation")
solver = cgraph.create("IterativeSolver")
pross = cgraph.create("AntennaPostprocess")


# 连接节点
mesher(cr = 0.05, sr0 = 1.9, sr1 = 2.4, L = 2.01, G = 0.01)
dipole_antenna_eq(
    mesh=mesher(),
    q=3,
    diffusion = pde().diffusion,
    reaction = pde().reaction,
    source = pde().source,
    Y = pde().Y,
    gd=pde().dirichlet,
)


solver(
    A = dipole_antenna_eq().A,
    b = dipole_antenna_eq().F,
    solver = "minres"
)

pross(uh = solver().out, mesh=mesher().mesh)

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher().mesh, uh=solver(), E =pross())

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

