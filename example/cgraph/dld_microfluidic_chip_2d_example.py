import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("DLDMicrofluidicChip2D")
mesher = cgraph.create("DLDMicrofluidicChipMesh2d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dld_eq = cgraph.create("DLDMicroflidicChipEquation")
solver = cgraph.create("DLDSolver")

uspacer(mesh = mesher(), p=2, gd = 2, value_dim = -1)
pspacer(mesh = mesher(), p=1)
pde(radius = mesher().radius,
    centers = mesher().centers,
    inlet_boundary = mesher().inlet_boundary,
    outlet_boundary = mesher().outlet_boundary,
    wall_boundary = mesher().wall_boundary)
dld_eq(uspace = uspacer(),
    pspace = pspacer(),
    velocity_dirichlet = pde().velocity_dirichlet, 
    pressure_dirichlet = pde().pressure_dirichlet, 
    is_velocity_boundary = pde().is_velocity_boundary, 
    is_pressure_boundary = pde().is_pressure_boundary)
solver(A = dld_eq().bform,
       b = dld_eq().lform,
       uspace = uspacer())

# 最终连接到图输出节点上
WORLD_GRAPH.output_node(uh = solver().uh)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
