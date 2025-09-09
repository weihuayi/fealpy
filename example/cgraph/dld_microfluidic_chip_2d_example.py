import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("DLDMicroflidicChip2D")

mesher = cgraph.create("ChipMesh2D")
isDDof = cgraph.create("BoundaryDof")
isDDof_p = cgraph.create("BoundaryDof")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dld_eq = cgraph.create("DLDMicroflidicChipEquation")
solver = cgraph.create("CGSolver")

uspacer(mesh = mesher(), p=2, gd = 2, value_dim = -1)
pspacer(mesh = mesher(), p=1)
dld_eq(uspace = uspacer(),
    pspace = pspacer(),
    velocity_dirichlet = pde().velocity_dirichlet, 
    pressure_dirichlet = pde().pressure_dirichlet, 
    is_velocity_boundary = pde().is_velocity_boundary, 
    is_pressure_boundary = pde().is_pressure_boundary)
solver(A = dld_eq().bform,
       b = dld_eq().lform)

# 最终连接到图输出节点上
WORLD_GRAPH.output_node(x = solver())
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
