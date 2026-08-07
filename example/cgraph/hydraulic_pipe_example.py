import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("ElbowPipeMesh")
show = cgraph.create("TO_VTK")

mesher(
    D=25.0,
    bend_angle=90.0,
    R_bend_inner=1.5,
    L_in_ratio=5.0,
    L_out_ratio=10.0,
    wall_thickness=10.0,
    mesh_size_global=7.5,
    mesh_size_bend=5,
    mesh_size_interface=3.75
    )

show(mesh=mesher().mesh, uh=None, path='C:/Users/Administrator/Desktop/新建文件夹 (2)')   

WORLD_GRAPH.output(mesh=mesher().mesh, show=show().path)

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())