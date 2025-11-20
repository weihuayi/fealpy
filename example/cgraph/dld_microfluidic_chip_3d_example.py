import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("DLDMicrofluidicChip3D")
mesher = cgraph.create("DLDMicrofluidicChipMesh3d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
dld_eq = cgraph.create("StokesEquation")
solver = cgraph.create("CGSolver")
postprocess = cgraph.create("VPDecoupling")
to_vtk = cgraph.create("TO_VTK")

mesher(lc = 0.07)
uspacer(mesh = mesher(), p=2, gd = 3)
pspacer(mesh = mesher(), p=1)
pde(thickness = mesher().thickness,
    radius = mesher().radius,
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
       b = dld_eq().lform)
postprocess(out = solver().out, 
            uspace = uspacer(),
            mesh = mesher())
to_vtk(mesh = mesher(),
        uh = (postprocess().uh, postprocess().ph),
        path = "/home/libz/dld_3d")

# 最终连接到图输出节点上
WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
