import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("StationaryNS2d")
mesher = cgraph.create("ChipMesh2D")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dld_eq = cgraph.create("DLDMicroflidicChipEquation")
dld_bc = cgraph.create("DLDMicrofluidicBC")

uspacer(mesh = mesher(), p=2, gd = 2, value_dim = -1)
pspacer(mesh = mesher(), p=1)
dld_eq(uspace = uspacer(),
    pspace = pspacer(),
    velocity_dirichlet = pde().velocity_dirichlet, 
    pressure_dirichlet = pde().pressure_dirichlet, 
    is_velocity_boundary = pde().is_velocity_boundary, 
    is_pressure_boundary = pde().is_pressure_boundary)



# 最终连接到图输出节点上
WORLD_GRAPH.output_node( dld_eq = dld_eq())

WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())