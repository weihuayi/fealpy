import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("TeePipeMesh")
show = cgraph.create("TO_VTK")

mesher(
    D_main=32.0,
    D_branch=32.0,
    intersect_angle=90.0,
    R_fillet=12.25,
    L_in_ratio=5.0,
    L_out_ratio=10.0,
    wall_thickness=5.0,
    mesh_size_global=6.0,
    mesh_size_junction=4.0,
    mesh_size_interface=3.0,
)

show(mesh=mesher().mesh, uh=None, path="./data")

WORLD_GRAPH.output(mesh=mesher().mesh, vtk_path=show().path)

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
